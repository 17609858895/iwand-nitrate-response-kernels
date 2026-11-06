"""Extend basin N surplus back to 1930 using county-scale TREND-Nitrogen (Byrnes et al., v3, CC BY 4.0).

1) area-weighted overlay of IWAND watershed polygons with the 2017 county boundaries (EPSG:5070);
2) basin series S_county(t) = sum_c A_ic * S_c(t) / sum_c A_ic for 1930-2017;
3) validation against the IWAND (gTREND-based) SURPLUS_N in 1980-2017;
4) splice: years < 1980 = a_i + b_i * S_county(t), with (a_i, b_i) fitted per basin on 1980-2017
   (b_i shrunk towards 1 when the overlap fit is weak), 1980-2023 kept as the official IWAND values.
Usage: python 13_build_long_surplus.py <Watershed_Boundary.gpkg>
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DP, RES  # noqa: E402

GPKG = sys.argv[1]
EXT = DP.parent / "data_external" / "TREND_nitrogen_extract"
cty = gpd.read_file(EXT / "county_2017" / "County_Boundaries_2017" / "Byrnesetal_TREND_CountyBoundaries.shp").to_crs(5070)
ws = gpd.read_file(GPKG).to_crs(5070)
ws["geometry"] = ws.geometry.simplify(100).buffer(0)
cty["geometry"] = cty.geometry.simplify(100).buffer(0)
print("loaded", len(ws), len(cty), flush=True)

WFILE = DP / "basin_county_weights.parquet"
sidx = cty.sindex
rows = []
if WFILE.exists():
    ws = ws.iloc[:0]  # reuse cached overlay weights
for i, (gid, geom) in enumerate(zip(ws["Gauge-ID"], ws.geometry)):
    cand = sidx.query(geom, predicate="intersects")
    if len(cand) == 0:
        continue
    inter = cty.geometry.iloc[cand].intersection(geom).area.to_numpy()
    for c, a in zip(cand, inter):
        if a > 0:
            rows.append((gid, cty["GEOID"].iloc[c], a))
    if i % 300 == 0:
        print(i, flush=True)
if rows:
    W = pd.DataFrame(rows, columns=["gid", "GEOID", "area_m2"])
    W["w"] = W["area_m2"] / W.groupby("gid")["area_m2"].transform("sum")
    W.to_parquet(WFILE, index=False)
W = pd.read_parquet(WFILE)

ns = pd.read_csv(EXT / "TREND-nitrogen" / "NSurplus.txt", dtype={"GEOID": str})
ycols = [c for c in ns.columns if c.startswith("y")]
S = ns.set_index("GEOID")[ycols]
S.columns = [int(c[1:]) for c in ycols]
M = W.merge(S, left_on="GEOID", right_index=True, how="left")
years = list(S.columns)
V = M[years].to_numpy(float)
wv = np.where(np.isnan(V), 0.0, M["w"].to_numpy()[:, None])
num = pd.DataFrame(np.nan_to_num(V) * wv, columns=years).groupby(M["gid"].to_numpy()).sum()
den = pd.DataFrame(wv, columns=years).groupby(M["gid"].to_numpy()).sum()
basin_cty = num / den
basin_cty.to_parquet(DP / "basin_surplus_trend_county_1930_2017.parquet")

# validation + splice
by = pd.read_parquet(DP / "basin_year.parquet")
iw = by.pivot(index="gid", columns="year", values="SURPLUS_N")
ov = list(range(1980, 2018))
stats, long = [], {}
for gid in iw.index:
    if gid not in basin_cty.index:
        continue
    a, b = iw.loc[gid, ov].to_numpy(float), basin_cty.loc[gid, ov].to_numpy(float)
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    slope, icpt = np.polyfit(b, a, 1)
    k = 0 if not np.isfinite(r) else max(min(r, 1), 0) ** 2  # shrink slope to 1 when overlap fit is weak
    slope_s = float(np.clip(k * slope + (1 - k) * 1.0, 0.5, 2.0))
    icpt_s = a.mean() - slope_s * b.mean()
    pre = icpt_s + slope_s * basin_cty.loc[gid, list(range(1930, 1980))].to_numpy(float)
    long[gid] = np.r_[pre, iw.loc[gid].to_numpy(float)]
    stats.append({"gid": gid, "r_overlap": r, "slope": slope, "slope_shrunk": slope_s, "mean_iwand": a.mean(),
                  "mean_county": b.mean(), "mean_1930_1979": pre.mean(), "mean_1950_1979": pre[20:].mean()})
st_ = pd.DataFrame(stats)
st_.to_csv(RES / "long_surplus_splice_diagnostics.csv", index=False)
cols = list(range(1930, 1980)) + list(iw.columns)
L = pd.DataFrame(long, index=cols).T
L.index.name = "gid"
L.to_parquet(DP / "basin_surplus_1930_2023_spliced.parquet")
print(st_[["r_overlap", "slope", "slope_shrunk", "mean_iwand", "mean_county", "mean_1930_1979"]].describe().round(3).to_string())
print("basins missing county overlay:", len(set(iw.index) - set(basin_cty.index)))
