"""Two-component survival-kernel DML (SK-DML) for rain -> river-nitrate impulse responses.

Treatment history: bin means of x_tau = log1p(rain + snowmelt) over 12 log-spaced lag bins (0-180 d).
Orthogonalisation: (basin x month) FE + year FE within-transformation; treatments linearly residualised on
within-transformed weather controls; outcome residualised by basin-grouped cross-fitted LightGBM.
Structural model (R-loss):
  y~ = beta_f(x) * sum_b m_b(mu_f(x), s_f) R~_b + beta_s(x) * sum_b m_b(mu_s(x), s_s) R~_b + gamma * F~ + e
  m_b = F(e_b + 1) - F(s_b), F = log-normal CDF over lag days (AFT survival: S = 1 - F)
  fast component: median 0.25-6 d (dilution / flushing); slow component: median 3-80 d (mobilisation)
Usage: python 22_rain_skdml.py [mode=real|placebo|synthetic] [period=all|1980-1999|2000-2011|2012-2023]
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

from common import DP, RES, load

torch.set_num_threads(12)
MODE = sys.argv[1] if len(sys.argv) > 1 else "real"
PERIOD = sys.argv[2] if len(sys.argv) > 2 else "all"
TAG = f"{MODE}_{PERIOD}"
rng = np.random.default_rng(0)
BINS = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 60), (61, 90), (91, 135), (136, 180)]
RCOLS = [f"R{a}_{b}" for a, b in BINS]
FCOLS = [f"FR{a}_{b}" for a, b in BINS]
CTRL = ["T7", "T30", "PET30", "SW7", "Q7", "SWE"]
ATTR = ["TOT_TILES_Early90s", "TOT_AGRICULTURE_2011", "TOT_FOREST_2011", "TOT_DEVELOPED_2011", "TOT_WETLANDS_2011",
        "TOT_BFI", "TOT_BASIN_SLOPE", "WSHDBDSqKM", "TOT_SANDAVE", "TOT_CLAYAVE", "TOT_PERMAVE", "TOT_WBM_PPT",
        "TOT_WBM_TAV", "TOT_WBM_SNW", "TOT_CORN_2023", "TOT_POPDENS10", "STR_QE_MA"]

d = pd.read_parquet(DP / "rain_bins.parquet")
_, st, _ = load()
if PERIOD != "all":
    a, b = map(int, PERIOD.split("-"))
    d = d[d["year"].between(a, b)]
hist_cols = FCOLS if MODE == "placebo" else RCOLS
d = d.dropna(subset=hist_cols + ["F1_30"] + CTRL).reset_index(drop=True)
d["m"] = pd.DatetimeIndex(d["Date"]).month
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


t0 = time.time()
Cw = within(d[CTRL].to_numpy())
Hw = within(d[hist_cols + ["F1_30"]].to_numpy())
Hw -= Cw @ np.linalg.lstsq(Cw, Hw, rcond=None)[0]  # linear residualisation of (exogenous) treatments on controls
A = st.set_index("gid").loc[gb_uni, ATTR].to_numpy(np.float64)
pos = np.nanmin(A, 0) >= 0
A = np.where(pos[None, :], np.log1p(np.clip(A, 0, None)), A)
A = np.nan_to_num((A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9))

# ------------------------------------------------------------------ synthetic truth (planted heterogeneous kernels)
z = lambda v: (v - v.mean()) / v.std()
TRUE = None
if MODE == "synthetic":
    from scipy.stats import norm
    ia = {c: i for i, c in enumerate(ATTR)}
    tb_s = 0.06 + 0.05 * A[:, ia["TOT_TILES_Early90s"]] + 0.03 * A[:, ia["TOT_AGRICULTURE_2011"]]
    tb_f = -0.03 - 0.02 * A[:, ia["TOT_BFI"]]
    tm_s = np.exp(np.log(12.0) + 0.4 * A[:, ia["TOT_BFI"]] - 0.3 * A[:, ia["TOT_TILES_Early90s"]])
    tm_f = np.full(len(A), 1.0)
    e_lo = np.array([max(a_, 1e-6) for a_, _ in BINS])
    e_hi = np.array([b_ + 1.0 for _, b_ in BINS])

    def masses(med, sig):
        return norm.cdf((np.log(e_hi)[None] - np.log(med)[:, None]) / sig) - norm.cdf((np.log(e_lo)[None] - np.log(med)[:, None]) / sig)
    Ms, Mf = masses(tm_s, 0.8), masses(tm_f, 0.6)
    sig = tb_s[gb_codes] * (Ms[gb_codes] * Hw[:, :12]).sum(1) + tb_f[gb_codes] * (Mf[gb_codes] * Hw[:, :12]).sum(1)
    TRUE = dict(beta_s=tb_s, beta_f=tb_f, med_s=tm_s)

# ------------------------------------------------------------------ outcome nuisance (cross-fitted LightGBM)
Yw = within(d[["y"]].to_numpy())[:, 0]
if MODE == "synthetic":
    Yw = sig + rng.normal(0, np.std(Yw) * 0.95, len(Yw))
NX = np.column_stack([Cw, d[["doy_s", "doy_c"]].to_numpy(), A[gb_codes]])
folds = list(GroupKFold(5).split(NX, groups=gb_codes))
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, n_jobs=12)
Yt = Yw.copy()
for tr, te in folds:
    Yt[te] = Yw[te] - lgb.train(P, lgb.Dataset(NX[tr], Yw[tr]), 300).predict(NX[te])
print(f"orthogonalisation {time.time() - t0:.0f}s; nuisance explains {1 - np.var(Yt) / np.var(Yw):.3f}", flush=True)

tY = torch.tensor(Yt, dtype=torch.float32)
tH = torch.tensor(Hw[:, :12], dtype=torch.float32)
tF = torch.tensor(Hw[:, 12], dtype=torch.float32)
tA = torch.tensor(A, dtype=torch.float32)
tg = torch.tensor(gb_codes, dtype=torch.long)
LO = torch.log(torch.tensor([max(a_, 1e-6) for a_, _ in BINS], dtype=torch.float32))
HI = torch.log(torch.tensor([b_ + 1.0 for _, b_ in BINS], dtype=torch.float32))
N01 = torch.distributions.Normal(0.0, 1.0)


def bin_mass(mu, sig):
    return N01.cdf((HI[None] - mu[:, None]) / sig[:, None]) - N01.cdf((LO[None] - mu[:, None]) / sig[:, None])


class Kernel2(nn.Module):
    """two_comp: fast+slow; regional_beta / regional_time: attribute linear (or MLP) indices."""

    def __init__(self, n_attr, two_comp=True, regional_beta=True, regional_time=True, mlp=False):
        super().__init__()
        self.two, self.rb, self.rt = two_comp, regional_beta, regional_time
        mk = (lambda k: nn.Sequential(nn.Linear(n_attr, 16), nn.Tanh(), nn.Linear(16, k))) if mlp else (lambda k: nn.Linear(n_attr, k))
        self.time_net, self.beta_net = mk(2), mk(2)
        for net in (self.time_net, self.beta_net):
            last = net[-1] if mlp else net
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, -0.5, 0.3, -0.01, 0.02, 0.0]))  # a_f, a_s, c_f, c_s, b_f, b_s, gamma

    def params(self, a):
        ht = self.time_net(a) if self.rt else torch.zeros(a.shape[0], 2)
        hb = self.beta_net(a) if self.rb else torch.zeros(a.shape[0], 2)
        mu_f = np.log(0.25) + (np.log(6.0) - np.log(0.25)) * torch.sigmoid(self.g[0] + ht[:, 0])
        mu_s = np.log(3.0) + (np.log(80.0) - np.log(3.0)) * torch.sigmoid(self.g[1] + ht[:, 1])
        s_f = 0.25 + 1.5 * torch.sigmoid(self.g[2]).expand(a.shape[0])
        s_s = 0.25 + 1.5 * torch.sigmoid(self.g[3]).expand(a.shape[0])
        b_f = self.g[4] + hb[:, 0]
        b_s = self.g[5] + hb[:, 1]
        if not self.two:
            b_f = b_f * 0
        return mu_f, mu_s, s_f, s_s, b_f, b_s

    def forward(self, idx):
        mu_f, mu_s, s_f, s_s, b_f, b_s = self.params(tA)
        g = tg[idx]
        contrib = b_s[g] * (bin_mass(mu_s, s_s)[g] * tH[idx]).sum(1) + b_f[g] * (bin_mass(mu_f, s_f)[g] * tH[idx]).sum(1)
        return contrib + self.g[6] * tF[idx]


def fit_kernel(cfg, tr, epochs=500, wd=1e-4):
    torch.manual_seed(0)
    m = Kernel2(A.shape[1], **cfg)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    ti = torch.from_numpy(tr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((tY[ti] - m(ti)) ** 2).mean() + wd * sum((p ** 2).sum() for n_, p in m.named_parameters() if n_ != "g")
        loss.backward()
        opt.step()
    return m


def kernel_sqerr(m, te):
    with torch.no_grad():
        ti = torch.from_numpy(te)
        return ((tY[ti] - m(ti)) ** 2).numpy()


Hn = Hw  # numpy treatments incl. future control


def dl_design(idx, inter):
    X = Hn[idx]
    if inter:
        X = np.column_stack([X] + [X[:, :12] * A[gb_codes[idx], j:j + 1] for j in range(A.shape[1])])
    return X


def fit_dl(tr, te, inter, ridge=1.0):
    Xtr, Xte = dl_design(tr, inter), dl_design(te, inter)
    coef = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ Yt[tr])
    return (Yt[te] - Xte @ coef) ** 2


MODELS = {
    "DL_bins_const": ("dl", False),
    "DL_bins_x_attributes": ("dl", True),
    "K1_global": ("k", dict(two_comp=False, regional_beta=False, regional_time=False)),
    "K2_global": ("k", dict(two_comp=True, regional_beta=False, regional_time=False)),
    "K2_regional_beta": ("k", dict(two_comp=True, regional_beta=True, regional_time=False)),
    "SK-DML_K2_regional": ("k", dict(two_comp=True, regional_beta=True, regional_time=True)),
    "SK-DML_K2_regional_mlp": ("k", dict(two_comp=True, regional_beta=True, regional_time=True, mlp=True)),
    "LGBM_flexible": ("lgbm", None),
}
E = {}
for name, (kind, cfg) in MODELS.items():
    e = np.zeros(len(Yt))
    for tr, te in folds:
        if kind == "dl":
            e[te] = fit_dl(tr, te, cfg)
        elif kind == "k":
            e[te] = kernel_sqerr(fit_kernel(cfg, tr), te)
        else:
            Xl = np.column_stack([Hn, A[gb_codes]])
            mdl = lgb.train(dict(P, num_leaves=31, min_data_in_leaf=1000), lgb.Dataset(Xl[tr], Yt[tr]), 300)
            e[te] = (Yt[te] - mdl.predict(Xl[te])) ** 2
    E[name] = e
    print(name, "done", f"{time.time() - t0:.0f}s", flush=True)

G = {k: np.bincount(gb_codes, v) for k, v in E.items()}
Gy = np.bincount(gb_codes, Yt ** 2)
draws = rng.integers(0, len(Gy), (300, len(Gy)))
rows = []
for k in MODELS:
    r2 = 1 - G[k].sum() / Gy.sum()
    r2b = 1 - G[k][draws].sum(1) / Gy[draws].sum(1)
    gain_dl = (G["DL_bins_const"][draws].sum(1) - G[k][draws].sum(1)) / Gy[draws].sum(1)
    gain_dlx = (G["DL_bins_x_attributes"][draws].sum(1) - G[k][draws].sum(1)) / Gy[draws].sum(1)
    rows.append({"model": k, "mode": MODE, "period": PERIOD, "oos_R2_pct": 100 * r2,
                 "lo": 100 * np.quantile(r2b, .025), "hi": 100 * np.quantile(r2b, .975),
                 "gain_vs_DLconst_pct": 100 * np.median(gain_dl), "gain_DLconst_lo": 100 * np.quantile(gain_dl, .025),
                 "gain_DLconst_hi": 100 * np.quantile(gain_dl, .975),
                 "gain_vs_DLxattr_pct": 100 * np.median(gain_dlx), "gain_DLxattr_lo": 100 * np.quantile(gain_dlx, .025),
                 "gain_DLxattr_hi": 100 * np.quantile(gain_dlx, .975)})
    print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items()}), flush=True)
pd.DataFrame(rows).to_csv(RES / f"rain_skdml_ablation_{TAG}.csv", index=False)

# ------------------------------------------------------------------ full fit, basin-level response functions
m = fit_kernel(MODELS["SK-DML_K2_regional"][1], np.arange(len(Yt)))
with torch.no_grad():
    mu_f, mu_s, s_f, s_s, b_f, b_s = m.params(tA)
    lags = torch.arange(0, 181, dtype=torch.float32)
    lo_ = torch.log(torch.clamp(lags, min=1e-6))
    hi_ = torch.log(lags + 1)
    pf = N01.cdf((hi_[None] - mu_f[:, None]) / s_f[:, None]) - N01.cdf((lo_[None] - mu_f[:, None]) / s_f[:, None])
    ps = N01.cdf((hi_[None] - mu_s[:, None]) / s_s[:, None]) - N01.cdf((lo_[None] - mu_s[:, None]) / s_s[:, None])
    h = (b_f[:, None] * pf + b_s[:, None] * ps).numpy()
kd = pd.DataFrame({"gid": gb_uni, "beta_fast": b_f.numpy(), "beta_slow": b_s.numpy(), "median_fast_d": np.exp(mu_f.numpy()),
                   "median_slow_d": np.exp(mu_s.numpy()), "sigma_fast": s_f.numpy(), "sigma_slow": s_s.numpy(),
                   "net_response": h.sum(1), "day0_3_response": h[:, :4].sum(1), "peak_day": np.argmax(h, 1),
                   "peak_value": h.max(1), "trough_value": h.min(1),
                   "gamma_future": float(m.g[6].detach())})
cdf_slow = np.cumsum(ps.numpy(), 1)
kd["t90_slow_d"] = (cdf_slow < 0.9).sum(1)
kd.to_csv(RES / f"rain_skdml_basin_params_{TAG}.csv", index=False)
np.save(RES / f"rain_skdml_response_curves_{TAG}.npy", h.astype(np.float32))
print(kd.describe().round(4).to_string(), flush=True)
attr_raw = st.set_index("gid").loc[gb_uni]
for c in ["TOT_TILES_Early90s", "TOT_AGRICULTURE_2011", "TOT_FOREST_2011", "TOT_BFI", "TOT_BASIN_SLOPE", "WSHDBDSqKM", "TOT_WBM_SNW"]:
    print(c, "spearman: beta_slow %.2f beta_fast %.2f median_slow %.2f net %.2f" % tuple(
        pd.Series(kd[k].values).corr(pd.Series(attr_raw[c].values), method="spearman")
        for k in ["beta_slow", "beta_fast", "median_slow_d", "net_response"]))
if TRUE is not None:
    print("recovery corr: beta_slow %.3f beta_fast %.3f log median_slow %.3f" % (
        np.corrcoef(TRUE["beta_s"], kd["beta_slow"])[0, 1], np.corrcoef(TRUE["beta_f"], kd["beta_fast"])[0, 1],
        np.corrcoef(np.log(TRUE["med_s"]), np.log(kd["median_slow_d"]))[0, 1]),
          "| median true/est slow median %.1f / %.1f d" % (np.median(TRUE["med_s"]), kd["median_slow_d"].median()), flush=True)
