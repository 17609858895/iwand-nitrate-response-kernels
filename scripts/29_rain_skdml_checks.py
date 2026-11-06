"""Supplementary checks for the A2 rain-response SK-DML (companion to 22_rain_skdml.py; same data, FE and folds).

Kernel family = AFT survival kernels  log T = mu(x) + s * eps,  eps ~ normal (log-normal, LN), logistic (log-logistic,
LL) or Gumbel-min (Weibull, WB), with mu the log median; s global, attribute-regionalised or fixed.
Curve-derived timing (robust to the mu / s trade-off): dominant-direction part of the daily response curve
h_dom = max(sign(net) * h, 0); t50 / t90 = days by which 50 / 90 % of it has arrived; peak = argmax h_dom.
Tasks (comma-separated):
  shapes  5-fold out-of-basin R2 + full fits for kernel families / sigma settings (linear regionalisation)
  timing  full fits only for the same variants (sub-period stability of timing quantities)
  boot    basin-cluster bootstrap (warm-started, lr 0.005) of the reference kernel -> CIs of curve-derived timing;
          group timing is read from the group-mean curve (ill-defined when a group mixes signs)
  state   state-dependent kernel beta(x) * (1 + kappa(x) * w); w = antecedent water-balance deficit 181-545 d before
          sampling (z within basin x month); placebo uses the deficit over the 365 d AFTER sampling
  synth   planted nonlinear, state-dependent kernels on the real sampling design; parameter and curve recovery
  cq      USGS sites with NWIS daily discharge only: rain kernel vs regionalised C-Q slope b(x) * log Q~(sampling day)
          vs both (out-of-basin), and full fits with / without the C-Q term (does delayed mobilisation survive?)
  holdout grab samples of the basins listed in env HOLDOUT (csv with column gid) are removed from the fit; their
          response curves are then predicted from attributes alone (for the nitrate-sensor reference comparison)
  eb      empirical-Bayes SK-DML: timing stays regional; per-basin amplitudes (fast, slow) from the basin's own grab
          samples are shrunk toward the regional amplitudes by precision. National fit, odd/even-year split prediction,
          and (with env HOLDOUT) curves for held-out basins whose data enter only the EB step
  ebsynth linear planted kernels + basin-specific amplitude deviations; recovery by prior / direct / EB (run alone)
Usage: python 29_rain_skdml_checks.py <period=all|1980-1999|...> <tasks> <climate cache dir>
Env QUICK=1: 150 basins, 30 epochs, 2 bootstrap draws (smoke test only).
Env PLACEBO=1: the rain history is replaced by the mirrored FUTURE bins (as in 22 placebo); outputs get a placebo_ tag.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from common import DP, RES

torch.set_num_threads(int(os.environ.get("NTHREADS", 12)))
PERIOD, TASKS, CACHE = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
assert not ({"synth", "ebsynth"} & set(TASKS)) or len(TASKS) == 1, "synthetic tasks replace the outcome; run them alone"
QUICK = os.environ.get("QUICK") == "1"
EPOCHS, NBOOT = (30, 2) if QUICK else (500, 40)
PLACEBO = os.environ.get("PLACEBO") == "1"
SUF = ("quick_" if QUICK else "") + ("placebo_" if PLACEBO else "") + PERIOD
rng = np.random.default_rng(0)
BINS = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 60), (61, 90), (91, 135), (136, 180)]
RCOLS = [f"FR{a}_{b}" for a, b in BINS] if PLACEBO else [f"R{a}_{b}" for a, b in BINS]
CTRL = ["T7", "T30", "PET30", "SW7", "Q7", "SWE"]
ATTR = ["TOT_TILES_Early90s", "TOT_AGRICULTURE_2011", "TOT_FOREST_2011", "TOT_DEVELOPED_2011", "TOT_WETLANDS_2011",
        "TOT_BFI", "TOT_BASIN_SLOPE", "WSHDBDSqKM", "TOT_SANDAVE", "TOT_CLAYAVE", "TOT_PERMAVE", "TOT_WBM_PPT",
        "TOT_WBM_TAV", "TOT_WBM_SNW", "TOT_CORN_2023", "TOT_POPDENS10", "STR_QE_MA"]
WIN = [(0, 3), (4, 10), (11, 30), (31, 90), (91, 180)]
NEED_W = bool({"state", "synth"} & set(TASKS))
NEED_Q = "cq" in TASKS
HOLD = pd.read_csv(os.environ["HOLDOUT"])["gid"].tolist() if ({"holdout", "eb"} & set(TASKS)) and "HOLDOUT" in os.environ else []
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ data (as in 22) + antecedent state
d = pd.read_parquet(DP / "rain_bins.parquet", columns=["gid", "Date", "y", "year"] + RCOLS + ["F1_30"] + CTRL + ["doy_s", "doy_c"])
st = pd.read_parquet(DP / "static.parquet")
if PERIOD != "all":
    a, b = map(int, PERIOD.split("-"))
    d = d[d["year"].between(a, b)]
if QUICK:
    keep = pd.Series(d["gid"].unique()).sample(150, random_state=0)
    d = d[d["gid"].isin(keep)]
d = d.dropna(subset=RCOLS + ["F1_30"] + CTRL)
d = (d[~d["gid"].isin(HOLD)] if "holdout" in TASKS else d).reset_index(drop=True)
d["m"] = pd.DatetimeIndex(d["Date"]).month
if NEED_W:
    gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
    cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
    ti = pd.date_range("1980-01-01", "2023-12-31", freq="D").get_indexer(pd.DatetimeIndex(d["Date"]))
    bi = d["gid"].map({g: i for i, g in enumerate(gids)}).to_numpy()
    wp, wf = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for b_ in np.unique(bi):  # per-basin cumulative water balance (P - PET), low memory
        r = np.where(bi == b_)[0]
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(cube[b_, :, 0].astype(np.float64) - cube[b_, :, 9]))])
        t = ti[r]
        ok = t - 545 >= 0
        wp[r[ok]] = -(cs[t[ok] - 180] - cs[t[ok] - 545])
        ok = t + 366 <= len(cs) - 1
        wf[r[ok]] = -(cs[t[ok] + 366] - cs[t[ok] + 1])
    d["w_past"], d["w_fut"] = wp, wf
    d = d.dropna(subset=["w_past", "w_fut"]).reset_index(drop=True)
    for c in ("w_past", "w_fut"):
        d[c] = d.groupby(["gid", "m"])[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9)).clip(-3, 3).fillna(0)
if NEED_Q:
    Qd = pd.read_parquet(DP / "nwis_dv.parquet")
    Qd["lq"] = np.log(Qd["Q_cfs"] + 0.01 * Qd.groupby("gid")["Q_cfs"].transform("median"))
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.merge(Qd[["gid", "Date", "lq"]], on=["gid", "Date"], how="inner").reset_index(drop=True)
gm = pd.factorize(d["gid"].astype(str) + "_" + d["m"].astype(str))[0]
gy = pd.factorize(d["year"])[0]
gb_codes, gb_uni = pd.factorize(d["gid"])


def within(a, n_iter=12):
    a = np.asarray(a, np.float64).copy()
    for _ in range(n_iter):
        for g in (gm, gy):
            s = np.zeros((g.max() + 1, a.shape[1]))
            np.add.at(s, g, a)
            a -= (s / np.bincount(g)[:, None])[g]
    return a


Cw = within(d[CTRL].to_numpy())
Hw = within(d[RCOLS + ["F1_30"]].to_numpy())
Hw -= Cw @ np.linalg.lstsq(Cw, Hw, rcond=None)[0]
A = st.set_index("gid").loc[gb_uni, ATTR].to_numpy(np.float64)
pos = np.nanmin(A, 0) >= 0
A = np.where(pos[None, :], np.log1p(np.clip(A, 0, None)), A)
A_MU, A_SD = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-9
A = np.nan_to_num((A - A_MU) / A_SD)


def attr_matrix(gid_list):
    """Attributes of any basins, transformed with the training basins' log / standardisation."""
    X = st.set_index("gid").loc[gid_list, ATTR].to_numpy(np.float64)
    X = np.where(pos[None, :], np.log1p(np.clip(X, 0, None)), X)
    return np.nan_to_num((X - A_MU) / A_SD)


ia = {c: i for i, c in enumerate(ATTR)}
attr_raw = st.set_index("gid").loc[gb_uni]
Yw = within(d[["y"]].to_numpy())[:, 0]
W = {k: d[k].to_numpy(np.float64) for k in ("w_past", "w_fut")} if NEED_W else {}
if NEED_Q:  # log discharge on the sampling day: within-transformed and residualised on controls like the treatments
    Qw = within(d[["lq"]].to_numpy())
    W["lq"] = (Qw - Cw @ np.linalg.lstsq(Cw, Qw, rcond=None)[0])[:, 0]
log("data", len(d), "rows", len(gb_uni), "basins")

# ------------------------------------------------------------------ planted truth (synth)
LAG = np.arange(181)
LO_D, HI_D = np.log(np.clip(LAG, 1e-6, None)), np.log(LAG + 1.0)
E_LO = np.log([max(a_, 1e-6) for a_, _ in BINS])
E_HI = np.log([b_ + 1.0 for _, b_ in BINS])
TRUE = None
if "synth" in TASKS:
    from scipy.stats import norm
    tile, ag, bfi = A[:, ia["TOT_TILES_Early90s"]], A[:, ia["TOT_AGRICULTURE_2011"]], A[:, ia["TOT_BFI"]]
    TRUE = dict(beta_s=0.01 + 0.10 / (1 + np.exp(-2.5 * (tile - 0.5))) + 0.03 * np.clip(ag, 0, None),
                beta_f=-0.03 - 0.025 * np.tanh(1.5 * bfi),
                med_s=np.exp(np.log(12.0) + 0.5 * np.tanh(bfi) - 0.4 / (1 + np.exp(-2 * tile))),
                med_f=np.full(len(A), 1.0), kappa_s=0.25 + 0.15 * np.tanh(ag), s_s=0.9, s_f=0.6)

    def ln_mass(med, s, lo, hi):
        return norm.cdf((hi[None] - np.log(med)[:, None]) / s) - norm.cdf((lo[None] - np.log(med)[:, None]) / s)
    Ms, Mf = ln_mass(TRUE["med_s"], TRUE["s_s"], E_LO, E_HI), ln_mass(TRUE["med_f"], TRUE["s_f"], E_LO, E_HI)
    g_ = gb_codes
    sig = (TRUE["beta_s"][g_] * (1 + TRUE["kappa_s"][g_] * W["w_past"]) * (Ms[g_] * Hw[:, :12]).sum(1)
           + TRUE["beta_f"][g_] * (Mf[g_] * Hw[:, :12]).sum(1))
    TRUE["curve"] = (TRUE["beta_s"][:, None] * ln_mass(TRUE["med_s"], TRUE["s_s"], LO_D, HI_D)
                     + TRUE["beta_f"][:, None] * ln_mass(TRUE["med_f"], TRUE["s_f"], LO_D, HI_D))
    Yw = sig + rng.normal(0, np.std(Yw) * 0.95, len(Yw))

if "ebsynth" in TASKS:  # linear attribute map + basin-specific amplitude deviations that attributes cannot explain
    from scipy.stats import norm
    tile, ag, bfi = A[:, ia["TOT_TILES_Early90s"]], A[:, ia["TOT_AGRICULTURE_2011"]], A[:, ia["TOT_BFI"]]
    u = rng.normal(0, 1, (len(A), 2))

    def lnm(med, s_, lo, hi):
        return norm.cdf((hi[None] - np.log(med)[:, None]) / s_) - norm.cdf((lo[None] - np.log(med)[:, None]) / s_)
    TRUE = dict(beta_s=0.06 + 0.05 * tile + 0.03 * ag + 0.05 * u[:, 1], beta_f=-0.03 - 0.02 * bfi + 0.02 * u[:, 0],
                med_s=np.exp(np.log(12.0) + 0.4 * bfi - 0.3 * tile), med_f=np.full(len(A), 1.0))
    Ms, Mf = lnm(TRUE["med_s"], 0.8, E_LO, E_HI), lnm(TRUE["med_f"], 0.6, E_LO, E_HI)
    sig = TRUE["beta_s"][gb_codes] * (Ms[gb_codes] * Hw[:, :12]).sum(1) + TRUE["beta_f"][gb_codes] * (Mf[gb_codes] * Hw[:, :12]).sum(1)
    TRUE["net"] = TRUE["beta_s"] * lnm(TRUE["med_s"], 0.8, LO_D, HI_D).sum(1) + TRUE["beta_f"] * lnm(TRUE["med_f"], 0.6, LO_D, HI_D).sum(1)
    Yw = sig + rng.normal(0, np.std(Yw) * 0.95, len(Yw))

# ------------------------------------------------------------------ outcome nuisance (cross-fitted LightGBM)
NX = np.column_stack([Cw, d[["doy_s", "doy_c"]].to_numpy(), A[gb_codes]] + ([within(d[["w_past", "w_fut"]].to_numpy())] if NEED_W else []))
folds = list(GroupKFold(5).split(NX, groups=gb_codes))
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, n_jobs=int(os.environ.get("NTHREADS", 12)))
Yt = Yw.copy()
for tr, te in folds:
    Yt[te] = Yw[te] - lgb.train(P, lgb.Dataset(NX[tr], Yw[tr]), 300).predict(NX[te])
log(f"orthogonalisation; nuisance explains {1 - np.var(Yt) / np.var(Yw):.3f}")
del NX

tY = torch.tensor(Yt, dtype=torch.float32)
tH = torch.tensor(Hw[:, :12], dtype=torch.float32)
tF = torch.tensor(Hw[:, 12], dtype=torch.float32)
tA = torch.tensor(A, dtype=torch.float32)
tg = torch.tensor(gb_codes, dtype=torch.long)
tW = {k: torch.tensor(v, dtype=torch.float32) for k, v in W.items()}
LO, HI = torch.tensor(E_LO, dtype=torch.float32), torch.tensor(E_HI, dtype=torch.float32)
N01 = torch.distributions.Normal(0.0, 1.0)
LLOG2 = float(np.log(np.log(2.0)))
CDF = {"LN": N01.cdf, "LL": torch.sigmoid, "WB": lambda u: 1 - torch.exp(-torch.exp(u + LLOG2))}  # all with median exp(mu)


class KernelX(nn.Module):
    """Two-component AFT kernel. sig: global | regional | fixed; state: None | 'w_past' | 'w_fut'."""

    def __init__(self, n_attr, fam="LN", sig="global", mlp=False, state=None, fix=(0.7, 1.2), cq=False):
        super().__init__()
        self.fam, self.sig, self.state, self.fix, self.cq = fam, sig, state, fix, cq
        if cq:  # regionalised C-Q slope (linear in attributes)
            self.cq_net = nn.Linear(n_attr, 1)
            nn.init.zeros_(self.cq_net.weight)
            nn.init.zeros_(self.cq_net.bias)
        mk = (lambda k: nn.Sequential(nn.Linear(n_attr, 16), nn.Tanh(), nn.Linear(16, k))) if mlp else (lambda k: nn.Linear(n_attr, k))
        self.time_net, self.beta_net = mk(4 if sig == "regional" else 2), mk(4 if state else 2)
        for net in (self.time_net, self.beta_net):
            last = net[-1] if mlp else net
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        # a_f, a_s, c_f, c_s, b_f, b_s, gamma, k_f, k_s
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, -0.5, 0.3, -0.01, 0.02, 0.0, 0.0, 0.0]))

    def params(self, a):
        ht, hb = self.time_net(a), self.beta_net(a)
        n = a.shape[0]
        p = dict(mu_f=np.log(0.25) + (np.log(6.0) - np.log(0.25)) * torch.sigmoid(self.g[0] + ht[:, 0]),
                 mu_s=np.log(3.0) + (np.log(80.0) - np.log(3.0)) * torch.sigmoid(self.g[1] + ht[:, 1]),
                 b_f=self.g[4] + hb[:, 0], b_s=self.g[5] + hb[:, 1])
        if self.sig == "fixed":
            p["s_f"], p["s_s"] = torch.full((n,), self.fix[0]), torch.full((n,), self.fix[1])
        else:
            rf, rs = (ht[:, 2], ht[:, 3]) if self.sig == "regional" else (0.0, 0.0)
            p["s_f"] = 0.25 + 1.5 * torch.sigmoid(self.g[2] + rf) * torch.ones(n)
            p["s_s"] = 0.25 + 1.5 * torch.sigmoid(self.g[3] + rs) * torch.ones(n)
        p["k_f"] = self.g[7] + hb[:, 2] if self.state else torch.zeros(n)
        p["k_s"] = self.g[8] + hb[:, 3] if self.state else torch.zeros(n)
        return p

    def mass(self, mu, s, lo, hi):
        F = CDF[self.fam]
        return F((hi[None] - mu[:, None]) / s[:, None]) - F((lo[None] - mu[:, None]) / s[:, None])

    def forward(self, idx):
        p = self.params(tA)
        g = tg[idx]
        cf = (self.mass(p["mu_f"], p["s_f"], LO, HI)[g] * tH[idx]).sum(1)
        cs = (self.mass(p["mu_s"], p["s_s"], LO, HI)[g] * tH[idx]).sum(1)
        bf, bs = p["b_f"][g], p["b_s"][g]
        if self.state:
            w = tW[self.state][idx]
            bf, bs = bf * (1 + p["k_f"][g] * w), bs * (1 + p["k_s"][g] * w)
        out = bf * cf + bs * cs + self.g[6] * tF[idx]
        if self.cq:
            out = out + self.cq_net(tA)[:, 0][g] * tW["lq"][idx]
        return out

    def curves(self, a=None):
        with torch.no_grad():
            p = self.params(tA if a is None else a)
            lo, hi = torch.tensor(LO_D, dtype=torch.float32), torch.tensor(HI_D, dtype=torch.float32)
            h = p["b_f"][:, None] * self.mass(p["mu_f"], p["s_f"], lo, hi) + p["b_s"][:, None] * self.mass(p["mu_s"], p["s_s"], lo, hi)
        return h.numpy().astype(np.float64), {k: v.numpy() for k, v in p.items()}


def fit_kernel(cfg, tr, epochs=EPOCHS, wd=1e-4, init=None, weight=None, lr=0.02):
    torch.manual_seed(0)
    m = KernelX(A.shape[1], **cfg)
    if init is not None:
        m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ti_ = torch.from_numpy(tr)
    wt = None if weight is None else torch.tensor(weight[tr], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        r2 = (tY[ti_] - m(ti_)) ** 2
        loss = (r2.mean() if wt is None else (wt * r2).sum() / wt.sum()) + wd * sum((p ** 2).sum() for n_, p in m.named_parameters() if n_ != "g")
        loss.backward()
        opt.step()
    return m


def kernel_sqerr(m, te):
    with torch.no_grad():
        ti_ = torch.from_numpy(te)
        return ((tY[ti_] - m(ti_)) ** 2).numpy()


def fit_dl(tr, te, inter, ridge=1.0):
    def design(idx):
        X = Hw[idx]
        return np.column_stack([X] + [X[:, :12] * A[gb_codes[idx], j:j + 1] for j in range(A.shape[1])]) if inter else X
    Xtr, Xte = design(tr), design(te)
    coef = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ Yt[tr])
    return (Yt[te] - Xte @ coef) ** 2


def fit_lgbm(tr, te, extra=()):
    Xl = np.column_stack([Hw, A[gb_codes]] + [W[k][:, None] for k in extra])
    mdl = lgb.train(dict(P, num_leaves=31, min_data_in_leaf=1000), lgb.Dataset(Xl[tr], Yt[tr]), 300)
    return (Yt[te] - mdl.predict(Xl[te])) ** 2


def fit_cq(tr, te, with_dl, ridge=1.0):
    """Closed-form regionalised C-Q slope (log Q~ x attributes), optionally plus DL x attributes."""
    def design(idx):
        q = W["lq"][idx][:, None]
        X = np.column_stack([q, q * A[gb_codes[idx]], Hw[idx, 12:13]])
        if with_dl:
            X = np.column_stack([X, Hw[idx, :12]] + [Hw[idx, :12] * A[gb_codes[idx], j:j + 1] for j in range(A.shape[1])])
        return X
    Xtr, Xte = design(tr), design(te)
    coef = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ Yt[tr])
    return (Yt[te] - Xte @ coef) ** 2


def cv(models):
    """models: name -> ('dl', inter) | ('k', cfg) | ('lgbm', extra) | ('oracle', None). Returns name -> sq. errors."""
    E = {}
    for name, (kind, cfg) in models.items():
        e = np.zeros(len(Yt))
        for tr, te in folds:
            if kind == "dl":
                e[te] = fit_dl(tr, te, cfg)
            elif kind == "k":
                e[te] = kernel_sqerr(fit_kernel(cfg, tr), te)
            elif kind == "cq":
                e[te] = fit_cq(tr, te, cfg)
            elif kind == "lgbm":
                e[te] = fit_lgbm(tr, te, cfg)
            else:
                e[te] = (Yt[te] - sig[te]) ** 2
        E[name] = e
        log(name, "done")
    return E


def summarise(E, ref, tag):
    G = {k: np.bincount(gb_codes, v) for k, v in E.items()}
    Gy = np.bincount(gb_codes, Yt ** 2)
    draws = rng.integers(0, len(Gy), (300, len(Gy)))
    rows = []
    for k in E:
        r2b = 1 - G[k][draws].sum(1) / Gy[draws].sum(1)
        gain = (G[ref][draws].sum(1) - G[k][draws].sum(1)) / Gy[draws].sum(1)
        rows.append({"model": k, "period": PERIOD, "oos_R2_pct": 100 * (1 - G[k].sum() / Gy.sum()),
                     "lo": 100 * np.quantile(r2b, .025), "hi": 100 * np.quantile(r2b, .975), "ref": ref,
                     "gain_pct": 100 * np.median(gain), "gain_lo": 100 * np.quantile(gain, .025), "gain_hi": 100 * np.quantile(gain, .975)})
        print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items()}), flush=True)
    pd.DataFrame(rows).to_csv(RES / f"rain_skdml_{tag}_ablation_{SUF}.csv", index=False)


def timing(h):
    """Curve-derived timing per basin from daily response curves (n_basin x 181)."""
    net = h.sum(1)
    hd = np.clip(np.sign(net)[:, None] * h, 0, None)
    tot = hd.sum(1, keepdims=True) + 1e-30
    C = np.cumsum(hd, 1) / tot
    out = {"net": net, "t50": (C < 0.5).sum(1), "t90": (C < 0.9).sum(1), "peak_dom": hd.argmax(1)}
    for a, b in WIN:
        out[f"share{a}_{b}"] = hd[:, a:b + 1].sum(1) / tot[:, 0]
    return pd.DataFrame(out)


GROUPS = {"all": np.ones(len(gb_uni), bool), "tile>10%": attr_raw["TOT_TILES_Early90s"].to_numpy() > 10,
          "ag>40%": attr_raw["TOT_AGRICULTURE_2011"].to_numpy() > 40, "ag<10%": attr_raw["TOT_AGRICULTURE_2011"].to_numpy() < 10}


def archetypes(h):
    """Archetype-level quantities from the mean response curve of each group."""
    r = {}
    for g, msk in GROUPS.items():
        if msk.sum() == 0:
            continue
        tm = timing(h[msk].mean(0, keepdims=True)).iloc[0]
        r.update({f"{g}|pct_mobilising": 100 * (h[msk].sum(1) > 0).mean(), f"{g}|net_mean": tm["net"],
                  f"{g}|t50": tm["t50"], f"{g}|t90": tm["t90"], f"{g}|peak": tm["peak_dom"]})
        r.update({f"{g}|share{a}_{b}": tm[f"share{a}_{b}"] for a, b in WIN})
    return r


def basin_table(m, variant):
    h, p = m.curves()
    t = timing(h)
    t.insert(0, "gid", gb_uni)
    t.insert(1, "variant", variant)
    for k in ("b_f", "b_s", "k_f", "k_s"):
        t[k] = p[k]
    t["median_fast_d"], t["median_slow_d"], t["sigma_fast"], t["sigma_slow"] = np.exp(p["mu_f"]), np.exp(p["mu_s"]), p["s_f"], p["s_s"]
    return t, h


ALL = np.arange(len(Yt))
VARIANTS = {"LN_global": dict(fam="LN"), "LN_sigma_regional": dict(fam="LN", sig="regional"),
            "LN_sigma_fixed": dict(fam="LN", sig="fixed"), "LL_global": dict(fam="LL"), "WB_global": dict(fam="WB")}

# ------------------------------------------------------------------ shapes / timing
ref_model = None
if {"shapes", "timing"} & set(TASKS):
    if "shapes" in TASKS:
        E = cv({"DL_bins_x_attributes": ("dl", True), **{k: ("k", v) for k, v in VARIANTS.items()}})
        summarise(E, "DL_bins_x_attributes", "shapes")
    tabs = []
    for k, v in VARIANTS.items():
        m = fit_kernel(v, ALL)
        if k == "LN_global":
            ref_model = m
        tabs.append(basin_table(m, k)[0])
        log("full fit", k)
    T = pd.concat(tabs)
    T.to_csv(RES / f"rain_skdml_shapes_basin_{SUF}.csv", index=False)
    piv = {q: T.pivot(index="gid", columns="variant", values=q) for q in ("median_slow_d", "t50", "peak_dom", "net")}
    for q, pv in piv.items():
        print(q, "Spearman vs LN_global:", pv.corr(method="spearman")["LN_global"].round(3).to_dict(), flush=True)
    print(T.groupby("variant")[["sigma_slow", "median_slow_d", "t50", "t90", "peak_dom"]].median().round(2).to_string(), flush=True)

# ------------------------------------------------------------------ bootstrap CIs of curve-derived timing
if "boot" in TASKS:
    if ref_model is None:
        ref_model = fit_kernel(VARIANTS["LN_global"], ALL)
    h0, _ = ref_model.curves()
    t0_, a0 = timing(h0), archetypes(h0)
    init = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}
    TB, AB = [], []
    for bdraw in range(NBOOT):
        cnt = np.bincount(rng.integers(0, len(gb_uni), len(gb_uni)), minlength=len(gb_uni))
        wobs = cnt[gb_codes].astype(np.float64)
        sel = np.where(wobs > 0)[0]
        mb = fit_kernel(VARIANTS["LN_global"], sel, epochs=EPOCHS // 2, init=init, weight=wobs, lr=0.005)
        hb, _ = mb.curves()
        TB.append(timing(hb).to_numpy())
        AB.append(archetypes(hb))
        log("boot", bdraw)
    TB = np.stack(TB)
    cols = t0_.columns.tolist()
    out = pd.DataFrame({"gid": gb_uni})
    for j, c in enumerate(cols):
        out[c] = t0_[c].to_numpy()
        out[c + "_lo"], out[c + "_hi"] = np.quantile(TB[:, :, j], 0.025, 0), np.quantile(TB[:, :, j], 0.975, 0)
    out["sign_support"] = (np.sign(TB[:, :, 0]) == np.sign(t0_["net"].to_numpy())[None]).mean(0)
    out.to_csv(RES / f"rain_skdml_boot_timing_{SUF}.csv", index=False)
    AB = pd.DataFrame(AB)
    arc = pd.DataFrame({"quantity": list(a0), "point": list(a0.values()),
                        "lo": AB.quantile(0.025).reindex(list(a0)).to_numpy(), "hi": AB.quantile(0.975).reindex(list(a0)).to_numpy()})
    arc.to_csv(RES / f"rain_skdml_boot_archetypes_{SUF}.csv", index=False)
    print(arc.round(3).to_string(index=False), flush=True)
    print("basins with net-response sign support >= 0.95: %.1f%%" % (100 * (out["sign_support"] >= 0.95).mean()), flush=True)
    print("median t50 CI width (days): %.1f" % np.median(out["t50_hi"] - out["t50_lo"]), flush=True)

# ------------------------------------------------------------------ state-dependent kernel (real) / planted truth (synth)
LIN, MLP = dict(fam="LN"), dict(fam="LN", mlp=True)


# ------------------------------------------------------------------ empirical-Bayes SK-DML (amplitudes partially pooled)
def kernel_parts(m):
    """Per-observation kernel-weighted rain histories (fast, slow) from the model's regional timing, regional
    amplitudes (prior), daily masses and the future-rain coefficient."""
    with torch.no_grad():
        p = m.params(tA)
        lo, hi = torch.tensor(LO_D, dtype=torch.float32), torch.tensor(HI_D, dtype=torch.float32)
        Mf, Ms = (m.mass(p[f"mu_{c}"], p[f"s_{c}"], LO, HI).numpy().astype(np.float64) for c in ("f", "s"))
        pf, ps = (m.mass(p[f"mu_{c}"], p[f"s_{c}"], lo, hi).numpy().astype(np.float64) for c in ("f", "s"))
    z = np.stack([(Mf[gb_codes] * Hw[:, :12]).sum(1), (Ms[gb_codes] * Hw[:, :12]).sum(1)], 1)
    prior = np.stack([p["b_f"].numpy(), p["b_s"].numpy()], 1).astype(np.float64)
    return z, prior, pf, ps, float(m.g[6].detach())


def eb_amplitudes(z, prior, gam, rows, min_n=20, n_iter=30):
    """Per-basin OLS amplitudes on `rows` (regional timing fixed; HC1 covariance V_i), shrunk toward the regional
    amplitudes: post_i = prior_i + T (T + V_i)^-1 (direct_i - prior_i). T (2x2 between-basin covariance of the
    deviations) by iterated precision-weighted moments, projected to PSD."""
    nb = len(gb_uni)
    g, zz = gb_codes[rows], z[rows]
    r = Yt[rows] - gam * Hw[rows, 12]
    n = np.bincount(g, minlength=nb)

    def bc(w):
        return np.bincount(g, w, minlength=nb)
    S = np.stack([[bc(zz[:, i] * zz[:, j]) for j in (0, 1)] for i in (0, 1)]).transpose(2, 0, 1)
    c = np.stack([bc(zz[:, i] * r) for i in (0, 1)], 1)
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2
    ok = (n >= min_n) & (det > 1e-6 * (S[:, 0, 0] * S[:, 1, 1] + 1e-30))
    Si = np.full((nb, 2, 2), np.nan)
    Si[ok] = np.linalg.inv(S[ok])
    direct = np.full((nb, 2), np.nan)
    direct[ok] = np.einsum("bij,bj->bi", Si[ok], c[ok])
    e = np.nan_to_num(r - (direct[g] * zz).sum(1))
    Mt = np.stack([[bc(e ** 2 * zz[:, i] * zz[:, j]) for j in (0, 1)] for i in (0, 1)]).transpose(2, 0, 1)
    V = np.full((nb, 2, 2), np.nan)
    V[ok] = np.einsum("bij,bjk,bkl->bil", Si[ok], Mt[ok], Si[ok]) * (n[ok] / (n[ok] - 2))[:, None, None]
    dl, Vo = direct[ok] - prior[ok], V[ok]
    T = np.cov(dl.T)
    for _ in range(n_iter):
        w = 1.0 / np.trace(T[None] + Vo, axis1=1, axis2=2) ** 2
        T = ((dl[:, :, None] * dl[:, None, :] - Vo) * w[:, None, None]).sum(0) / w.sum()
        ev, Q = np.linalg.eigh(T)
        T = (Q * np.clip(ev, 1e-10, None)) @ Q.T
    K = np.einsum("ij,bjk->bik", T, np.linalg.inv(T[None] + Vo))
    post = prior.copy()
    post[ok] = prior[ok] + np.einsum("bij,bj->bi", K, dl)
    wslow = np.zeros(nb)
    wslow[ok] = K[:, 1, 1]
    return dict(direct=direct, V=V, prior=prior, post=post, T=T, ok=ok, n=n, w_slow=wslow)


if "state" in TASKS or "synth" in TASKS:
    tag = "synth_nl" if "synth" in TASKS else "state"
    models = {"DL_bins_const": ("dl", False), "DL_bins_x_attributes": ("dl", True), "SK-DML_K2_regional": ("k", LIN),
              "SK-DML_K2_regional_mlp": ("k", MLP), "SK-DML_state_past": ("k", dict(LIN, state="w_past")),
              "SK-DML_mlp_state_past": ("k", dict(MLP, state="w_past")),
              "SK-DML_mlp_state_future_placebo": ("k", dict(MLP, state="w_fut")),
              "LGBM_flexible": ("lgbm", ()), "LGBM_+w_past": ("lgbm", ("w_past",)), "LGBM_+w_future": ("lgbm", ("w_fut",))}
    if tag == "synth_nl":
        models["oracle_truth"] = ("oracle", None)
    E = cv(models)
    summarise(E, "DL_bins_x_attributes", tag)
    rows, tabs = [], []
    for name, cfg in {"SK-DML_K2_regional": LIN, "SK-DML_K2_regional_mlp": MLP, "SK-DML_state_past": dict(LIN, state="w_past"),
                      "SK-DML_mlp_state_past": dict(MLP, state="w_past"), "SK-DML_state_future_placebo": dict(LIN, state="w_fut")}.items():
        m = fit_kernel(cfg, ALL)
        t, h = basin_table(m, name)
        tabs.append(t)
        r = {"model": name, "kappa_s_median": float(np.median(t["k_s"])), "kappa_f_median": float(np.median(t["k_f"]))}
        for c, n in (("TOT_TILES_Early90s", "tile"), ("TOT_AGRICULTURE_2011", "ag"), ("TOT_BFI", "BFI")):
            r[f"rho_kappa_s_{n}"] = pd.Series(t["k_s"].to_numpy()).corr(pd.Series(attr_raw[c].to_numpy()), method="spearman")
        for g, msk in GROUPS.items():
            r[f"kappa_s|{g}"] = float(np.median(t["k_s"].to_numpy()[msk])) if msk.any() else np.nan
        if TRUE is not None:
            tt = timing(TRUE["curve"])
            r.update({"rec_beta_s": np.corrcoef(TRUE["beta_s"], t["b_s"])[0, 1], "rec_beta_f": np.corrcoef(TRUE["beta_f"], t["b_f"])[0, 1],
                      "rec_log_med_s": np.corrcoef(np.log(TRUE["med_s"]), np.log(t["median_slow_d"]))[0, 1],
                      "rec_kappa_s": np.corrcoef(TRUE["kappa_s"], t["k_s"])[0, 1] if "state" in name and np.std(t["k_s"]) > 0 else np.nan,
                      "med_s_true_vs_est": f"{np.median(TRUE['med_s']):.1f}/{t['median_slow_d'].median():.1f}",
                      "curve_corr_median": float(np.median([np.corrcoef(TRUE["curve"][i], h[i])[0, 1] for i in range(len(h))]))})
            for q in ("net", "t50", "t90", "peak_dom", "share11_30"):
                r[f"rec_{q}_spearman"] = pd.Series(tt[q].to_numpy()).corr(pd.Series(t[q].to_numpy()), method="spearman")
            r["t50_true_vs_est"] = f"{np.median(tt['t50']):.0f}/{np.median(t['t50']):.0f}"
        rows.append(r)
        log("full fit", name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    pd.DataFrame(rows).to_csv(RES / f"rain_skdml_{tag}_fullfit_{SUF}.csv", index=False)
    pd.concat(tabs).to_csv(RES / f"rain_skdml_{tag}_basin_{SUF}.csv", index=False)
# ------------------------------------------------------------------ C-Q control (USGS sites with daily discharge)
if "cq" in TASKS:
    models = {"CQ_regional_slope": ("cq", False), "DL_bins_x_attributes": ("dl", True), "DL_x_attr+CQ": ("cq", True),
              "SK-DML_K2_regional": ("k", LIN), "SK-DML+CQ": ("k", dict(LIN, cq=True)),
              "SK-DML_K2_regional_mlp": ("k", MLP), "SK-DML_mlp+CQ": ("k", dict(MLP, cq=True)),
              "LGBM_flexible": ("lgbm", ()), "LGBM_+logQ": ("lgbm", ("lq",))}
    E = cv(models)
    summarise(E, "CQ_regional_slope", "cq")
    tabs, arc = [], []
    for name, cfg in {"SK-DML_K2_regional": LIN, "SK-DML+CQ": dict(LIN, cq=True)}.items():
        m = fit_kernel(cfg, ALL)
        t, h = basin_table(m, name)
        if m.cq:
            with torch.no_grad():
                t["cq_slope"] = m.cq_net(tA)[:, 0].numpy()
        tabs.append(t)
        arc.append({"model": name, **archetypes(h)})
        log("full fit", name)
    pd.concat(tabs).to_csv(RES / f"rain_skdml_cq_basin_{SUF}.csv", index=False)
    arc = pd.DataFrame(arc).set_index("model").T
    arc.to_csv(RES / f"rain_skdml_cq_archetypes_{SUF}.csv")
    print(arc.round(3).to_string(), flush=True)
    T2 = pd.concat(tabs).pivot(index="gid", columns="variant", values="net")
    print("net response Spearman with vs without C-Q term: %.3f" % T2.corr(method="spearman").iloc[0, 1], flush=True)
# ------------------------------------------------------------------ held-out basins: attribute-only predicted kernels
if "holdout" in TASKS:
    tAh = torch.tensor(attr_matrix(HOLD), dtype=torch.float32)
    rows = []
    for name, cfg in {"SK-DML_K2_regional": LIN, "SK-DML_K2_regional_mlp": MLP}.items():
        m = fit_kernel(cfg, ALL)
        h, p = m.curves(tAh)
        t = timing(h)
        t.insert(0, "gid", HOLD)
        t.insert(1, "model", name)
        t["b_f"], t["b_s"], t["median_slow_d"] = p["b_f"], p["b_s"], np.exp(p["mu_s"])
        rows.append(t)
        np.save(RES / f"rain_skdml_holdout_curves_{name}_{SUF}.npy", h.astype(np.float32))
        log("holdout fit", name, "basins in fit:", len(gb_uni), "held out:", len(HOLD))
    pd.concat(rows).to_csv(RES / f"rain_skdml_holdout_{SUF}.csv", index=False)
# ------------------------------------------------------------------ empirical-Bayes SK-DML: national, split, sensor, synthetic
if "eb" in TASKS or "ebsynth" in TASKS:
    tag = "ebsynth" if "ebsynth" in TASKS else "eb"
    hold_rows = d["gid"].isin(HOLD).to_numpy()
    PRIORS = {"LIN": LIN} if tag == "ebsynth" else {"LIN": LIN, "MLP": MLP}
    EST = ("prior (regional only)", "direct (own data; prior if n<20)", "EB-SK-DML")
    tabs, split_rows, rel_rows = [], [], []
    for pname, cfg in PRIORS.items():
        m = fit_kernel(cfg, np.where(~hold_rows)[0])  # the prior never sees the held-out (sensor) basins
        z, prior, pf, ps, gam = kernel_parts(m)
        eb = eb_amplitudes(z, prior, gam, ALL)
        Pf, Ps = pf.sum(1), ps.sum(1)

        def net(th):
            return th[:, 0] * Pf + th[:, 1] * Ps
        direct = np.where(eb["ok"][:, None], eb["direct"], np.nan)
        tab = pd.DataFrame({"gid": gb_uni, "prior": pname, "n": eb["n"], "ok": eb["ok"], "w_slow": eb["w_slow"],
                            "prior_bf": prior[:, 0], "prior_bs": prior[:, 1], "direct_bf": direct[:, 0], "direct_bs": direct[:, 1],
                            "direct_se_bs": np.sqrt(eb["V"][:, 1, 1]), "post_bf": eb["post"][:, 0], "post_bs": eb["post"][:, 1],
                            "prior_net": net(prior), "direct_net": net(direct), "post_net": net(eb["post"]),
                            "held_out": np.isin(np.asarray(gb_uni), HOLD)})
        if TRUE is not None:
            tab["true_bf"], tab["true_bs"], tab["true_net"] = TRUE["beta_f"], TRUE["beta_s"], TRUE["net"]
        tabs.append(tab)
        log(f"EB national ({pname}): T = {np.round(eb['T'], 5).tolist()}; basins with direct estimate {eb['ok'].sum()}; "
            f"median weight on own data (slow) {np.median(eb['w_slow'][eb['ok']]):.2f}")
        if len(HOLD):
            pos_ = {g_: i for i, g_ in enumerate(gb_uni)}
            hidx = [pos_[g_] for g_ in HOLD if g_ in pos_]
            for lab, th in (("prior", prior), ("eb", eb["post"])):
                np.save(RES / f"rain_skdml_eb_sensor_curves_{pname}_{lab}_{SUF}.npy",
                        (th[hidx, 0:1] * pf[hidx] + th[hidx, 1:2] * ps[hidx]).astype(np.float32))
        # temporal split: prior + EB fitted on odd (even) years predict the even (odd) years
        par = d["year"].to_numpy() % 2 == 0
        nets = {}
        for k, (fh, eh) in enumerate(((par, ~par), (~par, par))):
            rf, re_ = np.where(fh & ~hold_rows)[0], np.where(eh & ~hold_rows)[0]
            ms = fit_kernel(cfg, rf)
            zs_, pr_, pfs, pss, gms = kernel_parts(ms)
            ebs = eb_amplitudes(zs_, pr_, gms, rf)
            nf = np.bincount(gb_codes[rf], minlength=len(gb_uni))
            for lab, th in zip(EST, (pr_, np.where(ebs["ok"][:, None], ebs["direct"], pr_), ebs["post"])):
                pred = (th[gb_codes[re_]] * zs_[re_]).sum(1) + gms * Hw[re_, 12]
                e2 = np.bincount(gb_codes[re_], (Yt[re_] - pred) ** 2, len(gb_uni))
                y2 = np.bincount(gb_codes[re_], Yt[re_] ** 2, len(gb_uni))
                for sub, msk in (("all basins", np.ones(len(gb_uni), bool)), ("n_fit>=100", nf >= 100), ("n_fit<50", nf < 50)):
                    split_rows.append({"prior": pname, "split": k, "estimator": lab, "basins": sub,
                                       "R2_pct": 100 * (1 - e2[msk].sum() / y2[msk].sum())})
                nets[(k, lab)] = th[:, 0] * pfs.sum(1) + th[:, 1] * pss.sum(1)
            log(f"split {k} ({pname}) done; T = {np.round(ebs['T'], 5).tolist()}")
        for lab in EST[1:]:
            rel_rows.append({"prior": pname, "estimator": lab,
                             "spearman_net_odd_vs_even": pd.Series(nets[(0, lab)]).corr(pd.Series(nets[(1, lab)]), method="spearman")})
    T_ = pd.concat(tabs)
    T_.to_csv(RES / f"rain_skdml_{tag}_basin_{SUF}.csv", index=False)
    SP = pd.DataFrame(split_rows).groupby(["prior", "estimator", "basins"], as_index=False)["R2_pct"].mean()
    SP.to_csv(RES / f"rain_skdml_{tag}_split_{SUF}.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(RES / f"rain_skdml_{tag}_reliability_{SUF}.csv", index=False)
    pd.set_option("display.width", 220)
    print(SP.round(3).to_string(index=False), flush=True)
    print(pd.DataFrame(rel_rows).round(3).to_string(index=False), flush=True)
    for pname, t_ in T_.groupby("prior"):
        t_ = t_.set_index("gid")
        print(f"[{pname}] group mean net (prior -> EB):", {g_: (round(t_["prior_net"].to_numpy()[msk].mean(), 3), round(t_["post_net"].to_numpy()[msk].mean(), 3))
                                                         for g_, msk in GROUPS.items()}, flush=True)
        print(f"[{pname}] sign flips prior -> EB: {100 * (np.sign(t_['prior_net']) != np.sign(t_['post_net'])).mean():.1f}%", flush=True)
        if TRUE is not None:
            for col in ("prior", "direct", "post"):
                v = t_.dropna(subset=[f"{col}_net"])
                print(f"[{pname}] truth recovery {col:6s}: corr beta_s {np.corrcoef(v[f'{col}_bs'], v['true_bs'])[0, 1]:.3f}  "
                      f"RMSE beta_s {np.sqrt(np.mean((v[f'{col}_bs'] - v['true_bs']) ** 2)):.4f}  "
                      f"corr net {np.corrcoef(v[f'{col}_net'], v['true_net'])[0, 1]:.3f}  (n={len(v)})", flush=True)
log("finished")
