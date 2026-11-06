"""Are SK-DML timescales short because of the data or the estimator? Pooled nonparametric distributed lags (12 bins,
basin x month + year FE, same construction as 36_sensor_validation.py) on IWAND grab samples, compared with the dense
sensor truth and with the SK-DML archetype curves (rain_skdml_boot_archetypes_all.csv)."""
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(__file__).with_name("36_sensor_validation.py").read_text(encoding="utf-8").split("P = pd.read_parquet")[0]
ns = {"__file__": str(Path(__file__).with_name("36_sensor_validation.py")), "__name__": "sv"}
exec(compile(src, "36_head", "exec"), ns)
dl_fit, curve, timing, RC, CTRL, DP, RES = (ns[k] for k in ("dl_fit", "curve", "timing", "RC", "CTRL", "DP", "RES"))

grab = pd.read_parquet(DP / "rain_bins.parquet", columns=["gid", "Date", "y", "year"] + RC + ["F1_30"] + CTRL)
st = pd.read_parquet(DP / "static.parquet").set_index("gid")
sens = pd.read_parquet(DP / "sensor_daily_panel.parquet")
sites = sens["gid"].unique()
span = sens.groupby("gid")["Date"].agg(["min", "max"])
g_in = grab[grab["gid"].isin(sites)]
g_sp = g_in[g_in.apply(lambda r: span.loc[r["gid"], "min"] <= pd.Timestamp(r["Date"]) <= span.loc[r["gid"], "max"], axis=1)]
tile = grab["gid"].map(st["TOT_TILES_Early90s"]) > 10
ag = grab["gid"].map(st["TOT_AGRICULTURE_2011"]) > 40
rows = []
for lab, df in [("sensor sites: dense sensor series", sens), ("sensor sites: IWAND grab, all years", g_in),
                ("sensor sites: IWAND grab, sensor period", g_sp), ("all tile>10% basins: IWAND grab", grab[tile]),
                ("all ag>40% basins: IWAND grab", grab[ag])]:
    c, se, net, net_se, n = dl_fit(df, site_fe=True)
    rows.append({"data": lab, "n": n, "basins": df["gid"].nunique(), "net_se": net_se, **timing(curve(c))})
    print(lab, n, flush=True)
arc = pd.read_csv(RES / "rain_skdml_boot_archetypes_all.csv").set_index("quantity")
for g in ("tile>10%", "ag>40%"):
    rows.append({"data": f"SK-DML archetype curve {g} (bootstrap point)", "net": arc.loc[f"{g}|net_mean", "point"],
                 "t50": arc.loc[f"{g}|t50", "point"], "peak": arc.loc[f"{g}|peak", "point"],
                 **{f"share{a}_{b}": arc.loc[f"{g}|share{a}_{b}", "point"] for a, b in ns["WIN"]}})
T = pd.DataFrame(rows)
T.to_csv(RES / "timing_bias_check.csv", index=False)
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False))

# timing by basin size within tile / agricultural basins (pooled DL)
area = grab["gid"].map(st["WSHDBDSqKM"])
rows = []
for lab, msk in [("tile>10%", tile), ("ag>40%", ag)]:
    for a_lab, lo, hi in [("<1,000 km2", 0, 1e3), ("1,000-10,000 km2", 1e3, 1e4), (">10,000 km2", 1e4, np.inf)]:
        df = grab[msk & area.between(lo, hi, inclusive="left")]
        if df["gid"].nunique() < 10:
            continue
        c, se, net, net_se, n = dl_fit(df, site_fe=True)
        rows.append({"group": lab, "area": a_lab, "basins": df["gid"].nunique(), "n": n, "net_se": net_se, **timing(curve(c))})
A = pd.DataFrame(rows)
A.to_csv(RES / "timing_by_basin_size.csv", index=False)
print(A.round(3).to_string(index=False))
print("sensor-site median area (km2): %.0f" % st.loc[sites, "WSHDBDSqKM"].median())
