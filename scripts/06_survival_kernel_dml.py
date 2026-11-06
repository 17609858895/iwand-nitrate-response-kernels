"""Survival-kernel double machine learning (SK-DML) for legacy-nitrogen response timing.

Panel unit: basin-year (obs <= 2017, >= 4 samples/yr). Outcome: annual mean of deseasonalised log nitrate.
Treatment history: R_tau = log1p(SURPLUS_{t-tau}), tau = 0..40 (pre-1980 held at 1980 value).
Model:   y~_it = beta(x_i) * sum_tau w_tau(x_i) * R~_tau,it + e_it
         w_tau(x) = S(tau|x) - S(tau+1|x),  log T = mu(x) + sigma(x) * eps   (AFT survival kernel)
Orthogonalisation: two-way (basin, year) within-transformation, then cross-fitted LightGBM residualisation of
y and of every R_tau on within-transformed climate + static attributes (basin-grouped folds).
Usage: python 06_survival_kernel_dml.py [family] [mode=real|synthetic] [maxyear=2017] [final model] [trend=1]
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from common import RES, load, n_history, static_features

torch.set_num_threads(8)
FAMILY = sys.argv[1] if len(sys.argv) > 1 else "lognormal"
MODE = sys.argv[2] if len(sys.argv) > 2 else "real"
MAXYEAR = int(sys.argv[3]) if len(sys.argv) > 3 else 2017
TREND = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # 1 = also remove basin-specific linear trends
LONG = int(sys.argv[6]) if len(sys.argv) > 6 else 0  # 1 = TREND-Nitrogen-spliced 1930 history
TAG = f"{FAMILY}_{MODE}" + ("" if MAXYEAR == 2017 else f"_to{MAXYEAR}") + ("" if TREND else "_notrend") + ("_long" if LONG else "")
MAXLAG = 40
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- panel
obs, st, by = load()
obs = obs[obs["year"] <= MAXYEAR].copy()
obs["m"] = obs["Date"].dt.month
obs["ya"] = obs["y"] - obs.groupby(["gid", "m"])["y"].transform("mean")
ccols = ["P30", "P90", "W30", "T30", "PET30", "WB90", "MELT30", "API97"]
an = obs.groupby(["gid", "year"]).agg(ya=("ya", "mean"), n=("ya", "size"),
                                       **{f"s_{c}": (c, "mean") for c in ccols}).reset_index()
an = an[an["n"] >= 4].merge(by[["gid", "year", "P", "T", "PET", "MELT", "P_PET"]], on=["gid", "year"])
an = an.reset_index(drop=True)
SF = static_features(st)
Xs = st.set_index("gid")[SF]
hist, _ = n_history(by, "SURPLUS_N", MAXLAG, long=bool(LONG))
R = np.stack([np.log1p(np.clip(hist[l].stack().reindex(pd.MultiIndex.from_arrays([an.gid, an.year])).to_numpy(), 0, None))
              for l in range(MAXLAG + 1)], 1)
CLIM = ["P", "T", "PET", "MELT", "P_PET"] + [f"s_{c}" for c in ccols]
gid_codes, gid_uni = pd.factorize(an["gid"])
yr_codes, _ = pd.factorize(an["year"])


TT = (an["year"].to_numpy() - 1998) / 10.0


def within(a, n_iter=15):
    """Basin FE + year FE (+ basin-specific linear trends if TREND) by alternating projections; a: [n, k]."""
    a = a.astype(np.float64).copy()
    for _ in range(n_iter):
        for codes in (gid_codes, yr_codes):
            s = np.zeros((codes.max() + 1, a.shape[1]))
            np.add.at(s, codes, a)
            cnt = np.bincount(codes)[:, None]
            a -= (s / cnt)[codes]
        if TREND:
            sxx = np.bincount(gid_codes, TT * TT)
            sxy = np.zeros((gid_codes.max() + 1, a.shape[1]))
            np.add.at(sxy, gid_codes, a * TT[:, None])
            a -= (sxy / np.maximum(sxx, 1e-9)[:, None])[gid_codes] * TT[:, None]
    return a


Yw = within(an[["ya"]].to_numpy())[:, 0]
Rw = within(R)
Cw = within(an[CLIM].to_numpy())
S_attr = Xs.loc[an["gid"]].to_numpy(np.float64)

# ---------------------------------------------------------------- synthetic truth (identifiability test)
attr = st.set_index("gid")
z = lambda v: (v - v.mean()) / v.std()
true_med = np.exp(np.log(4.0) + 0.6 * z(attr["TOT_BFI"]) + 0.3 * z(np.log1p(attr["TOT_AGRICULTURE_2011"])))
true_beta = 0.15 + 0.25 / (1 + np.exp(-1.5 * z(np.log1p(attr["TOT_AGRICULTURE_2011"]))))
true_sig = 0.6
if MODE == "synthetic":
    from scipy.stats import norm
    edges = np.log(np.arange(1, MAXLAG + 1))
    med_i = true_med.loc[an["gid"]].to_numpy()
    c = norm.cdf((edges[None, :] - np.log(med_i)[:, None]) / true_sig)
    w = np.concatenate([c[:, :1], np.diff(c, axis=1), 1 - c[:, -1:]], 1)
    signal = true_beta.loc[an["gid"]].to_numpy() * (w * R).sum(1)
    clim_conf = 0.15 * np.tanh(z(an["P_PET"]).to_numpy())  # climate also drives y
    noise = rng.normal(0, np.std(Yw) * 0.9, len(an))
    y_raw = signal + clim_conf + noise + an.groupby("gid")["ya"].transform("mean").to_numpy()
    Yw = within(y_raw[:, None])[:, 0]

# ---------------------------------------------------------------- cross-fitted nuisance residualisation
NUIS_X = np.column_stack([Cw, S_attr])
folds = list(GroupKFold(5).split(NUIS_X, groups=gid_codes))
P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=8)


def crossfit(target, rounds=300):
    out = np.zeros(len(target))
    for tr, te in folds:
        m = lgb.train(P, lgb.Dataset(NUIS_X[tr], target[tr]), rounds)
        out[te] = m.predict(NUIS_X[te])
    return target - out


t0 = time.time()
cache = RES.parent / "data_processed" / f"skdml_cache_to{MAXYEAR}_trend{TREND}_long{LONG}.npz"
if cache.exists():
    cz = np.load(cache)
    Rt = cz["Rt"]
    Yt = crossfit(Yw) if MODE == "synthetic" else cz["Yt"]
else:
    Yt = crossfit(Yw)
    Rt = np.column_stack([crossfit(Rw[:, l], 150) for l in range(MAXLAG + 1)])
    if MODE == "real":
        np.savez_compressed(cache, Yt=Yt, Rt=Rt, Yw=Yw)
nuis_r2 = 1 - np.var(Yt) / np.var(Yw)
print(f"nuisance done {time.time() - t0:.0f}s; climate explains {nuis_r2:.3f} of within-variance", flush=True)

# ---------------------------------------------------------------- SK-DML estimator
Xstd = Xs.to_numpy(np.float64)
pos = np.nanmin(Xstd, 0) >= 0
Xstd = np.where(pos[None, :], np.log1p(np.clip(Xstd, 0, None)), Xstd)
Xstd = np.nan_to_num((Xstd - np.nanmean(Xstd, 0)) / (np.nanstd(Xstd, 0) + 1e-6)).astype(np.float32)
tX = torch.from_numpy(Xstd)
tb = torch.from_numpy(gid_codes.astype(np.int64))
gid_to_row = {g: i for i, g in enumerate(st["gid"])}
row_of_unit = torch.from_numpy(np.array([gid_to_row[g] for g in gid_uni]))
tY = torch.from_numpy(Yt.astype(np.float32))
tR = torch.from_numpy(Rt.astype(np.float32))
LOGE = torch.log(torch.arange(1, MAXLAG + 1, dtype=torch.float32))
EDGE = torch.arange(1, MAXLAG + 1, dtype=torch.float32)


def surv(edges_log, mu, sig, fam):
    zz = (edges_log[None, :] - mu[:, None]) / sig[:, None]
    if fam == "lognormal":
        return 1 - torch.distributions.Normal(0.0, 1.0).cdf(zz)
    if fam == "weibull":  # log T ~ Gumbel(min)
        return torch.exp(-torch.exp(zz))
    if fam == "loglogistic":
        return 1 - torch.sigmoid(zz)
    raise ValueError(fam)


class SKDML(nn.Module):
    def __init__(self, nx, hetero_beta=True, hetero_lag=True, lag=True, fam="lognormal", arch="mlp"):
        super().__init__()
        self.hb, self.hl, self.lag, self.fam = hetero_beta, hetero_lag, lag, fam
        mk = (lambda k: nn.Sequential(nn.Linear(nx, 16), nn.Tanh(), nn.Linear(16, k))) if arch == "mlp" else             (lambda k: nn.Sequential(nn.Linear(nx, k)))
        self.lag_net, self.beta_net = mk(2), mk(1)  # separate heads: timing vs magnitude
        self.net = nn.ModuleList([self.lag_net, self.beta_net])
        for head in (self.lag_net, self.beta_net):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.g = nn.Parameter(torch.tensor([np.log(4.0), np.log(0.7), -1.0]))

    def params(self, x):
        hl, hb = self.lag_net(x), self.beta_net(x)[:, 0]
        mu = self.g[0] + (hl[:, 0] if self.hl else 0 * hl[:, 0])
        sig = torch.exp(self.g[1] + (0.5 * torch.tanh(hl[:, 1]) if self.hl else 0 * hl[:, 1]))
        beta = nn.functional.softplus(self.g[2] + (hb if self.hb else 0 * hb))
        return torch.clamp(mu, np.log(0.25), np.log(60.0)), torch.clamp(sig, 0.15, 2.5), beta

    def weights(self, x):
        mu, sig, beta = self.params(x)
        if not self.lag:
            w = torch.zeros(x.shape[0], MAXLAG + 1)
            w[:, 0] = 1
            return w, beta, mu, sig
        Sv = surv(LOGE, mu, sig, self.fam)
        w = torch.cat([1 - Sv[:, :1], Sv[:, :-1] - Sv[:, 1:], Sv[:, -1:]], 1)
        return w, beta, mu, sig

    def forward(self, idx):
        rows = row_of_unit[tb[idx]]
        w, beta, _, _ = self.weights(tX[rows])
        return beta * (w * tR[idx]).sum(1)


def train(cfg, tr_idx, epochs=300, wd=1e-3):
    torch.manual_seed(0)
    cfg = dict(cfg)
    wd = cfg.pop("wd", wd)
    m = SKDML(tX.shape[1], **cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=0)
    ti = torch.from_numpy(tr_idx)
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((tY[ti] - m(ti)) ** 2).mean() + wd * sum((p ** 2).sum() for p in m.net.parameters())
        loss.backward()
        opt.step()
    return m


def sqerr(m, idx):
    with torch.no_grad():
        ti = torch.from_numpy(idx)
        return ((tY[ti] - m(ti)) ** 2).numpy()


CFGS = {
    "TWFE_lag0_const": dict(hetero_beta=False, hetero_lag=False, lag=False),
    "lag0_hetero_beta": dict(hetero_beta=True, hetero_lag=False, lag=False),
    "global_kernel_const": dict(hetero_beta=False, hetero_lag=False, lag=True),
    "global_kernel_hetero_beta": dict(hetero_beta=True, hetero_lag=False, lag=True),
    "SK-DML_full": dict(hetero_beta=True, hetero_lag=True, lag=True),
    "SK-DML_linear": dict(hetero_beta=True, hetero_lag=True, lag=True, arch="linear", wd=3e-3),
}
base = float((tY ** 2).mean())
rows, E = [], {}
y2 = Yt ** 2
for name, cfg in CFGS.items():
    cfg = dict(cfg, fam=FAMILY)
    e = np.zeros(len(Yt))
    for tr, te in GroupKFold(5).split(Yt, groups=gid_codes):
        e[te] = sqerr(train(cfg, tr), te)
    E[name] = e
# basin-cluster bootstrap of out-of-basin explained within-variance and paired gain vs TWFE lag-0
B = 300
units = np.unique(gid_codes)
grp_sum = lambda v: np.bincount(gid_codes, v)
G0, Gy = {k: grp_sum(v) for k, v in E.items()}, grp_sum(y2)
draws = rng.integers(0, len(units), (B, len(units)))
for name in CFGS:
    r2 = 1 - G0[name].sum() / Gy.sum()
    r2b = 1 - G0[name][draws].sum(1) / Gy[draws].sum(1)
    d = (G0["TWFE_lag0_const"][draws].sum(1) - G0[name][draws].sum(1)) / Gy[draws].sum(1)
    rows.append({"model": name, "family": FAMILY, "mode": MODE, "oos_R2": r2, "oos_R2_lo": np.quantile(r2b, .025),
                 "oos_R2_hi": np.quantile(r2b, .975), "gain_vs_lag0": (G0["TWFE_lag0_const"].sum() - G0[name].sum()) / Gy.sum(),
                 "gain_lo": np.quantile(d, .025), "gain_hi": np.quantile(d, .975)})
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
pd.DataFrame(rows).to_csv(RES / f"skdml_ablation_{TAG}.csv", index=False)

# full fit + basin-level parameters
FINAL = sys.argv[4] if len(sys.argv) > 4 else "SK-DML_linear"
m = train(dict(CFGS[FINAL], fam=FAMILY), np.arange(len(Yt)))
with torch.no_grad():
    w, beta, mu, sig = m.weights(tX)
    cdf = torch.cumsum(w, 1).numpy()
q50 = (cdf < 0.5).sum(1)
q90 = (cdf < 0.9).sum(1)
kd = pd.DataFrame({"gid": st["gid"], "beta": beta.numpy(), "median_lag": np.exp(mu.numpy()), "sigma": sig.numpy(),
                   "t50_yr": q50, "t90_yr": q90})
kd.to_csv(RES / f"skdml_basin_params_{TAG}.csv", index=False)
print(kd.describe().round(3).to_string(), flush=True)
if MODE == "synthetic":
    tm, tbeta = true_med.loc[st["gid"]].to_numpy(), true_beta.loc[st["gid"]].to_numpy()
    print("recovery: corr(log median lag)", np.corrcoef(np.log(tm), np.log(kd["median_lag"]))[0, 1],
          "corr(beta)", np.corrcoef(tbeta, kd["beta"])[0, 1], "median true/est lag", np.median(tm), kd["median_lag"].median())
else:
    for c in ["TOT_BFI", "TOT_AGRICULTURE_2011", "TOT_TILES_Early90s", "TOT_BASIN_SLOPE", "TOT_SANDAVE", "TOT_WBM_PPT"]:
        print(c, "spearman with median lag %.2f, with beta %.2f" % (
            pd.Series(kd["median_lag"].values).corr(pd.Series(st[c].values), method="spearman"),
            pd.Series(kd["beta"].values).corr(pd.Series(st[c].values), method="spearman")))
