"""Decision-gate tests: does a 1930-2017 surplus history make longer lags identifiable?

For two histories (short: IWAND 1980-, pre-1980 held at 1980; long: TREND-Nitrogen spliced from 1930),
basin-year panel (<= 2017, >= 4 samples/yr), outcome = annual mean deseasonalised log nitrate:
  - lag-kernel profile (log-normal, sigma 0.6, median 0.5-40 yr) with basin+year FE, with/without basin trends
  - future-surplus placebo (years <= 2012) with/without basin trends
  - class profiles (BFI terciles, agriculture, tile drainage, sand) with basin trends
Clustered (basin) SEs; SSE used to locate the best kernel.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import RES, load, surplus_wide

MAXLAG = 80
MEDS = [0.5, 1, 2, 3, 5, 8, 12, 16, 20, 30, 40]
obs, st, by = load()
obs = obs[obs["year"] <= 2017].copy()
obs["m"] = obs["Date"].dt.month
obs["ya"] = obs["y"] - obs.groupby(["gid", "m"])["y"].transform("mean")
an = obs.groupby(["gid", "year"]).agg(ya=("ya", "mean"), n=("ya", "size")).reset_index()
an = an[an["n"] >= 4].merge(by[["gid", "year", "P", "T", "PET", "MELT"]], on=["gid", "year"]).reset_index(drop=True)
attr = st.set_index("gid")
an = an.join(attr[["TOT_BFI", "TOT_AGRICULTURE_2011", "TOT_TILES_Early90s", "TOT_SANDAVE"]], on="gid")
g1_all, g2_all = pd.factorize(an["gid"])[0], pd.factorize(an["year"])[0]
TT_all = (an["year"].to_numpy() - 1998) / 10.0
edges = np.log(np.arange(1, MAXLAG + 1))


def weights(med, sig=0.6):
    c = norm.cdf((edges - np.log(med)) / sig)
    return np.r_[c[0], np.diff(c), 1 - c[-1]]


def kernel_panel(wide, med, lead=False):
    lp = np.log1p(np.clip(wide.to_numpy(float), 0, None))
    yrs = wide.columns.to_numpy()
    W = weights(med)
    out = np.zeros_like(lp)
    T = lp.shape[1]
    for l, w in enumerate(W):
        idx = np.clip(np.arange(T) + (1 + l if lead else -l), 0, T - 1)
        out += w * lp[:, idx]
    return pd.DataFrame(out, index=wide.index, columns=yrs).stack().reindex(
        pd.MultiIndex.from_arrays([an["gid"], an["year"]])).to_numpy()


def ols(L, trend, mask=None):
    mask = np.ones(len(an), bool) if mask is None else np.asarray(mask)
    D = np.column_stack([an["ya"], L, an[["P", "T", "PET", "MELT"]]])[mask].astype(float)
    a1, a2, tt = pd.factorize(g1_all[mask])[0], pd.factorize(g2_all[mask])[0], TT_all[mask]
    for _ in range(15):
        for g in (a1, a2):
            s = np.zeros((g.max() + 1, D.shape[1]))
            np.add.at(s, g, D)
            D -= (s / np.bincount(g)[:, None])[g]
        if trend:
            sxx = np.bincount(a1, tt * tt)
            sxy = np.zeros((a1.max() + 1, D.shape[1]))
            np.add.at(sxy, a1, D * tt[:, None])
            D -= (sxy / np.maximum(sxx, 1e-9)[:, None])[a1] * tt[:, None]
    y, X = D[:, 0], D[:, 1:]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    U = np.zeros((a1.max() + 1, X.shape[1]))
    np.add.at(U, a1, X * e[:, None])
    se = np.sqrt((XtXi @ (U.T @ U) @ XtXi)[0, 0])
    return b[0], se, float(e @ e)


q = an["TOT_BFI"].quantile([1 / 3, 2 / 3]).to_numpy()
CLASSES = {"all": np.ones(len(an), bool), "BFI_low": an["TOT_BFI"] < q[0],
           "BFI_mid": (an["TOT_BFI"] >= q[0]) & (an["TOT_BFI"] < q[1]), "BFI_high": an["TOT_BFI"] >= q[1],
           "ag>40%": an["TOT_AGRICULTURE_2011"] > 40, "ag<10%": an["TOT_AGRICULTURE_2011"] < 10,
           "tile>10%": an["TOT_TILES_Early90s"] > 10, "sand>40%": an["TOT_SANDAVE"] > 40}
prof, plac, cls = [], [], []
le = an["year"].to_numpy() <= 2012
for hist in ["short", "long"]:
    wide = surplus_wide(by, long=(hist == "long"))
    if hist == "short":  # pre-1980 held at the 1980 value, as in the original analysis
        for y in range(1930, 1980):
            wide[y] = wide[1980]
        wide = wide[sorted(wide.columns)]
    K = {m: kernel_panel(wide, m) for m in MEDS}
    for trend in (False, True):
        for m in MEDS:
            b, se, sse = ols(K[m], trend)
            prof.append({"history": hist, "trend": trend, "median_lag": m, "beta": b, "se": se, "t": b / se, "sse": sse})
        for m in (5, 12):
            b, se, _ = ols(kernel_panel(wide, m, lead=True), trend, le)
            b0, se0, _ = ols(K[m], trend, le)
            plac.append({"history": hist, "trend": trend, "median_lag": m, "lagged_beta": b0, "lagged_t": b0 / se0,
                         "future_beta": b, "future_t": b / se})
    for name, mask in CLASSES.items():
        res = {m: ols(K[m], True, mask) for m in MEDS}
        best = min(res, key=lambda m: res[m][2])
        for m, (b, se, sse) in res.items():
            cls.append({"history": hist, "class": name, "median_lag": m, "beta": b, "se": se, "t": b / se, "sse": sse,
                        "best": m == best})
        print(hist, name, "best median lag", best, "beta %.3f t %.1f" % (res[best][0], res[best][0] / res[best][1]), flush=True)
P = pd.DataFrame(prof)
P.to_csv(RES / "long_history_lag_profiles.csv", index=False)
pd.DataFrame(plac).to_csv(RES / "long_history_placebo.csv", index=False)
pd.DataFrame(cls).to_csv(RES / "long_history_class_profiles.csv", index=False)
for (h, tr), d in P.groupby(["history", "trend"]):
    best = d.loc[d["sse"].idxmin()]
    print(f"{h:5s} trend={tr!s:5s} best median {best.median_lag:>4} beta {best.beta:.3f} t {best.t:.1f} |",
          " ".join(f"{r.median_lag:g}:{r.beta:.2f}({r.t:.1f})" for r in d.itertuples()))
print(pd.DataFrame(plac).round(3).to_string(index=False))
