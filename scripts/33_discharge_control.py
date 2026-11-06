"""Is the rain -> nitrate response just a concentration-discharge (C-Q) relation? USGS sites with NWIS daily discharge.

Same lag-bin design as 20_rain_response_lags.py (rain bins z-scored within basin x month; basin + year FE on the
basin x month nitrate anomaly; basin-clustered t). Per group:
  A  y ~ rain bins                              (baseline on the discharge subset)
  B  y ~ rain bins + log Q (sampling day)       (instantaneous C-Q control)
  C  y ~ rain bins + log Q day 0 + Q lag bins   (C-Q with flow memory / hysteresis)
  D  log Q (sampling day) ~ rain bins           (hydrograph response on the same days)
  E  y ~ log Q (sampling day)                   (C-Q only)
Per basin: C-Q slope b (elasticity of nitrate to daily discharge anomaly) vs the SK-DML net rain response.
Usage: python 33_discharge_control.py <climate cache dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(__file__).with_name("20_rain_response_lags.py").read_text(encoding="utf-8").split("cols = list(BINS)")[0]
ns = {"__file__": str(Path(__file__).with_name("20_rain_response_lags.py")), "__name__": "rl"}
exec(compile(src, "20_head", "exec"), ns)
obs, z, attr, RES, bi, ti, NT, BINS = (ns[k] for k in ("obs", "z", "attr", "RES", "bi", "ti", "NT", "BINS"))
DP = RES.parent / "data_processed"
RAIN = [c for c in BINS if c != "F1_30"]

# ------------------------------------------------------------------ daily log discharge matrix
Q = pd.read_parquet(DP / "nwis_dv.parquet")
dates = pd.date_range("1980-01-01", "2023-12-31", freq="D")
sites = Q["gid"].unique()
si = {g: i for i, g in enumerate(sites)}
LQ = np.full((len(sites), NT), np.nan)
for g, q in Q.groupby("gid"):
    pos = dates.get_indexer(q["Date"])
    ok = pos >= 0
    v = q["Q_cfs"].to_numpy()
    LQ[si[g], pos[ok]] = np.log(v[ok] + 0.01 * np.median(v))
cs = np.concatenate([np.zeros((len(sites), 1)), np.cumsum(np.nan_to_num(LQ), 1)], 1)
cn = np.concatenate([np.zeros((len(sites), 1)), np.cumsum(np.isfinite(LQ), 1)], 1)
obs = obs[obs["gid"].isin(si)].copy()
qi = obs["gid"].map(si).to_numpy()
t = pd.Series(ti, index=ns["obs"].index).loc[obs.index].to_numpy()  # ti is positional on the unfiltered obs
obs["lq_raw"] = np.where((t >= 0) & (t < NT), LQ[qi, np.clip(t, 0, NT - 1)], np.nan)
QB = {"QL1_3": (1, 3), "QL4_10": (4, 10), "QL11_30": (11, 30), "QL31_90": (31, 90), "QL91_180": (91, 180)}
for name, (a, b) in QB.items():
    lo, hi = t - b, t - a + 1
    ok = (lo >= 0) & (hi <= NT)
    s_, n_ = np.full(len(t), np.nan), np.full(len(t), np.nan)
    s_[ok] = cs[qi[ok], hi[ok]] - cs[qi[ok], lo[ok]]
    n_[ok] = cn[qi[ok], hi[ok]] - cn[qi[ok], lo[ok]]
    obs[name] = np.where(n_ >= 0.5 * (b - a + 1), s_ / np.maximum(n_, 1), np.nan)
obs["lq_anom"] = obs["lq_raw"] - obs.groupby(["gid", "m"])["lq_raw"].transform("mean")
for c in ["lq_raw"] + list(QB):
    obs[c + "_z" if c == "lq_raw" else c] = obs.groupby(["gid", "m"])[c].transform(z)
obs = obs.rename(columns={"lq_raw_z": "lq0"})
d = obs.replace([np.inf, -np.inf], np.nan).dropna(subset=RAIN + ["lq0", "ya"] + list(QB)).copy()
print(f"discharge subset: {len(d):,} samples, {d['gid'].nunique()} basins", flush=True)


def fe(dd, ycol, cols):
    """basin + year FE OLS with basin-clustered SE; returns b, t, SSE, SST (within)."""
    D = dd[[ycol] + cols].to_numpy(float).copy()
    g1, g2 = pd.factorize(dd["gid"])[0], pd.factorize(dd["year"])[0]
    for _ in range(10):
        for g in (g1, g2):
            s = np.zeros((g.max() + 1, D.shape[1]))
            np.add.at(s, g, D)
            D -= (s / np.bincount(g)[:, None])[g]
    y, X = D[:, 0], D[:, 1:]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    U = np.zeros((g1.max() + 1, X.shape[1]))
    np.add.at(U, g1, X * e[:, None])
    se = np.sqrt(np.diag(XtXi @ (U.T @ U) @ XtXi))
    return b, b / se, float(e @ e), float(y @ y)


A_ = d["gid"].map(attr["TOT_AGRICULTURE_2011"]).to_numpy()
T_ = d["gid"].map(attr["TOT_TILES_Early90s"]).to_numpy()
B_ = d["gid"].map(attr["TOT_BFI"]).to_numpy()
qb = np.nanquantile(attr["TOT_BFI"], [1 / 3, 2 / 3])
GROUPS = {"all": np.ones(len(d), bool), "tile>10%": T_ > 10, "ag>40%": A_ > 40, "ag<10%": A_ < 10, "BFI_high": B_ >= qb[1]}
SPECS = {"A_rain": ("ya", RAIN), "B_rain+Q0": ("ya", RAIN + ["lq0"]), "C_rain+Qhist": ("ya", RAIN + ["lq0"] + list(QB)),
         "D_Q0~rain": ("lq0", RAIN), "E_Q0_only": ("ya", ["lq0"]), "F_Qhist_only": ("ya", ["lq0"] + list(QB))}
rows = []
for grp, msk in GROUPS.items():
    dg = d[msk]
    for spec, (ycol, cols) in SPECS.items():
        b, tt, sse, sst = fe(dg, ycol, cols)
        rows.append({"group": grp, "spec": spec, "n": len(dg), "basins": dg["gid"].nunique(), "R2_within_pct": 100 * (1 - sse / sst),
                     **{f"{c}_b": v for c, v in zip(cols, b)}, **{f"{c}_t": v for c, v in zip(cols, tt)}})
        print(f"{grp:9s} {spec:14s} R2w {100 * (1 - sse / sst):5.2f}% ", " ".join(f"{c}:{x:+.3f}({y:.1f})" for c, x, y in zip(cols, b, tt) if c in RAIN + ["lq0"]), flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "discharge_control_lag_bins.csv", index=False)

# ------------------------------------------------------------------ per-basin C-Q slope vs SK-DML net rain response
cq = d.groupby("gid").apply(lambda g: pd.Series({"cq_slope": np.cov(g["ya"], g["lq_anom"])[0, 1] / (np.var(g["lq_anom"], ddof=1) + 1e-12),
                                                  "n": len(g)}), include_groups=False)
cq = cq[cq["n"] >= 30]
kd = pd.read_csv(RES / "rain_skdml_basin_params_real_all.csv").set_index("gid")
bt = pd.read_csv(RES / "rain_skdml_boot_timing_all.csv").set_index("gid")
cq = cq.join(kd[["net_response", "beta_slow", "beta_fast"]]).join(bt[["sign_support"]])
cq["cq_class"] = pd.cut(cq["cq_slope"], [-np.inf, -0.1, 0.1, np.inf], labels=["dilution", "chemostatic", "enrichment"])
cq["kernel_sign"] = np.where(cq["net_response"] > 0, "mobilising", "diluting")
cq.to_csv(RES / "discharge_cq_vs_kernel_by_basin.csv")
print(f"\nbasins with >=30 samples: {len(cq)}")
print("Spearman C-Q slope vs net response %.2f, vs beta_slow %.2f, vs beta_fast %.2f" % tuple(
    cq["cq_slope"].corr(cq[c], method="spearman") for c in ("net_response", "beta_slow", "beta_fast")))
print(pd.crosstab(cq["cq_class"], cq["kernel_sign"]).to_string())
hs = cq["sign_support"] >= 0.95
print("high-support basins only:")
print(pd.crosstab(cq.loc[hs, "cq_class"], cq.loc[hs, "kernel_sign"]).to_string())
