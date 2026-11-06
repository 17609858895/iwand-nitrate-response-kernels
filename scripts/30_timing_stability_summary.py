"""Which timing quantities are identified? Cross-period and cross-family agreement of the log-normal median parameter
versus curve-derived timing (t50, peak), from rain_skdml_shapes_basin_<period>.csv written by 29_rain_skdml_checks.py."""
from itertools import combinations

import numpy as np
import pandas as pd

from common import DP, RES

PERIODS = ["1980-1999", "2000-2011", "2012-2023"]
Q = ["median_slow_d", "t50", "t90", "peak_dom", "net", "b_s"]
st = pd.read_parquet(DP / "static.parquet").set_index("gid")
T = {p: pd.read_csv(RES / f"rain_skdml_shapes_basin_{p}.csv") for p in PERIODS + ["all"]}

rows = []
for v in T["all"]["variant"].unique():
    for a, b in combinations(PERIODS, 2):
        ta, tb = (T[p][T[p]["variant"] == v].set_index("gid") for p in (a, b))
        g = ta.index.intersection(tb.index)
        rows.append({"variant": v, "pair": f"{a} vs {b}", "n": len(g),
                     **{q: ta.loc[g, q].corr(tb.loc[g, q], method="spearman") for q in Q}})
C = pd.DataFrame(rows)
C.to_csv(RES / "timing_stability_cross_period.csv", index=False)

rows = []
for p, t in T.items():
    for v, tv in t.groupby("variant"):
        tv = tv.set_index("gid")
        mob = tv["net"] > 0
        tile = st.loc[tv.index, "TOT_TILES_Early90s"] > 10
        rows.append({"period": p, "variant": v, "sigma_slow": tv["sigma_slow"].median(), "median_slow_d": tv["median_slow_d"].median(),
                     "t50_mobilising": tv.loc[mob, "t50"].median(), "t90_mobilising": tv.loc[mob, "t90"].median(),
                     "peak_mobilising": tv.loc[mob, "peak_dom"].median(), "t50_tile": tv.loc[tile & mob, "t50"].median(),
                     "pct_mobilising": 100 * mob.mean()})
L = pd.DataFrame(rows)
L.to_csv(RES / "timing_stability_levels.csv", index=False)

pd.set_option("display.width", 200)
print(C.groupby("variant")[Q].mean().round(3).to_string())
print(L.round(2).to_string(index=False))
