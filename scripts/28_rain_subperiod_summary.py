"""Sub-period stability of the A2 rain-response kernels (reads outputs of 22_rain_skdml.py).
Archetype summaries per period, attribute rank correlations, and cross-period basin-level agreement."""
from itertools import combinations

import numpy as np
import pandas as pd

from common import DP, RES

PERIODS = ["all", "1980-1999", "2000-2011", "2012-2023"]
st = pd.read_parquet(DP / "static.parquet").set_index("gid")
ATT = {"TOT_TILES_Early90s": "tile", "TOT_AGRICULTURE_2011": "ag", "TOT_FOREST_2011": "forest", "TOT_BFI": "BFI"}
WIN = [(0, 3), (4, 10), (11, 30), (31, 90), (91, 180)]

kd, H = {}, {}
for p in PERIODS:
    k = pd.read_csv(RES / f"rain_skdml_basin_params_real_{p}.csv")
    kd[p] = k.join(st[list(ATT)], on="gid")
    H[p] = pd.DataFrame(np.load(RES / f"rain_skdml_response_curves_real_{p}.npy"), index=k["gid"])

rows = []
for p in PERIODS:
    k, h = kd[p], H[p]
    groups = {"all": np.ones(len(k), bool), "tile>10%": k["TOT_TILES_Early90s"] > 10, "ag>40%": k["TOT_AGRICULTURE_2011"] > 40,
              "ag<10%": k["TOT_AGRICULTURE_2011"] < 10}
    for g, m in groups.items():
        m = np.asarray(m)
        pos = m & (k["net_response"].to_numpy() > 0)
        hm = h.to_numpy()[m].mean(0)
        r = {"period": p, "group": g, "n": int(m.sum()), "pct_mobilising": 100 * pos.sum() / m.sum(),
             "peak_day_med_pos": float(np.median(k["peak_day"].to_numpy()[pos])) if pos.any() else np.nan,
             "median_slow_d": float(np.median(k["median_slow_d"].to_numpy()[m])), "net_mean": float(k["net_response"][m].mean())}
        for a, b in WIN:
            r[f"H{a}_{b}"] = 1000 * hm[a:b + 1].sum()
        rows.append(r)
    for c, n in ATT.items():
        rows.append({"period": p, "group": f"rho_{n}", **{v: k[v].corr(k[c], method="spearman")
                                                          for v in ("beta_slow", "beta_fast", "net_response", "median_slow_d")}})
S = pd.DataFrame(rows)
S.to_csv(RES / "rain_skdml_subperiod_summary.csv", index=False)

rows = []
for a, b in combinations(PERIODS, 2):
    ka, kb = kd[a].set_index("gid"), kd[b].set_index("gid")
    g = ka.index.intersection(kb.index)
    r = {"pair": f"{a} vs {b}", "n_common": len(g)}
    for v in ("beta_slow", "beta_fast", "net_response", "peak_day", "median_slow_d"):
        r[v] = ka.loc[g, v].corr(kb.loc[g, v], method="spearman")
    ha, hb = H[a].loc[g].to_numpy(), H[b].loc[g].to_numpy()
    ha, hb = ha - ha.mean(1, keepdims=True), hb - hb.mean(1, keepdims=True)
    cc = (ha * hb).sum(1) / np.sqrt((ha ** 2).sum(1) * (hb ** 2).sum(1) + 1e-30)
    r["curve_corr_median"] = float(np.median(cc))
    r["sign_agree_pct"] = 100 * float((np.sign(ka.loc[g, "net_response"]) == np.sign(kb.loc[g, "net_response"])).mean())
    rows.append(r)
C = pd.DataFrame(rows)
C.to_csv(RES / "rain_skdml_subperiod_agreement.csv", index=False)
pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))
print(C.round(3).to_string(index=False))
