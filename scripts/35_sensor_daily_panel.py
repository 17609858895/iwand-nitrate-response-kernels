"""Daily panel for USGS continuous nitrate-sensor days at IWAND basins (dense reference series).

Same rain-history bins as 21_rain_history_bins.py (bin means of log1p(rain + degree-day melt) over 12 lag bins and the
future 1-30 d bin) and the same weather controls as 01_build_features.py, computed for EVERY sensor day; days with an
IWAND grab sample at the same site are flagged (y_grab). Output: data_processed/sensor_daily_panel.parquet
Usage: python 35_sensor_daily_panel.py <climate cache dir>
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

from common import DP, EPS

BINS = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 60), (61, 90), (91, 135), (136, 180)]
CACHE = sys.argv[1]
DATES = pd.date_range("1980-01-01", "2023-12-31", freq="D")
MIN_DAYS = 365


@njit(cache=True)
def snow(p, t, ddf=3.0):
    """Degree-day snow bucket: liquid water input (rain + melt) and SWE (mm/day), as in 01 / 21."""
    w, swe = np.zeros(p.shape[0]), np.zeros(p.shape[0])
    s = 0.0
    for i in range(p.shape[0]):
        rain = p[i] if t[i] > 0.0 else 0.0
        if t[i] <= 0.0:
            s += p[i]
        m = 0.0
        if t[i] > 0.0 and s > 0.0:
            m = min(s, ddf * t[i])
            s -= m
        w[i], swe[i] = rain + m, s
    return w, swe


def trailing(x, w, how="sum"):
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    s = np.full(x.shape[0], np.nan)
    s[w - 1:] = c[w:] - c[:-w]
    return s if how == "sum" else s / w


S = pd.read_parquet(DP / "nwis_sensor_no3.parquet")
gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
gi = {g: i for i, g in enumerate(gids)}
cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
grab = pd.read_parquet(DP / "rain_bins.parquet", columns=["gid", "Date", "y"])
grab = grab.assign(Date=pd.to_datetime(grab["Date"])).groupby(["gid", "Date"], as_index=False)["y"].mean()
out = []
for g, s in S.groupby("gid"):
    if g not in gi or len(s) < MIN_DAYS:
        continue
    c = np.asarray(cube[gi[g]], dtype=np.float64)  # (days, 11): APCP, DLWRF, DSWRF, PRES, TMP, UGRD, CONVfrac, VGRD, SPFH, PEVAP, CAPE
    p, sw, t, q, pet = (np.nan_to_num(c[:, k]) for k in (0, 2, 4, 8, 9))
    w, swe = snow(p, t)
    cs = np.concatenate([[0.0], np.cumsum(np.log1p(w))])
    ti = DATES.get_indexer(pd.DatetimeIndex(s["Date"]))
    s, ti = s[ti >= 0], ti[ti >= 0]
    rec = {"gid": g, "Date": s["Date"].to_numpy(), "y": np.log(s["NO3_mgL"].to_numpy() + EPS)}
    for a, b in BINS:
        lo, hi = ti - b, ti - a + 1
        v = np.full(len(ti), np.nan)
        ok = lo >= 0
        v[ok] = (cs[hi[ok]] - cs[lo[ok]]) / (hi[ok] - lo[ok])
        rec[f"R{a}_{b}"] = v
    v = np.full(len(ti), np.nan)
    ok = ti + 31 <= len(w)
    v[ok] = (cs[ti[ok] + 31] - cs[ti[ok] + 1]) / 30
    rec["F1_30"] = v
    ctrl = {"T7": trailing(t, 7, "mean"), "T30": trailing(t, 30, "mean"), "PET30": trailing(pet, 30),
            "SW7": trailing(sw, 7, "mean"), "Q7": trailing(q, 7, "mean"), "SWE": swe}
    rec.update({k: v_[ti] for k, v_ in ctrl.items()})
    doy = pd.DatetimeIndex(s["Date"]).dayofyear.to_numpy()
    rec["doy_s"], rec["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out.append(pd.DataFrame(rec))
Pn = pd.concat(out, ignore_index=True)
Pn["Date"] = pd.to_datetime(Pn["Date"])
Pn["year"] = Pn["Date"].dt.year
Pn = Pn.merge(grab.rename(columns={"y": "y_grab"}), on=["gid", "Date"], how="left")
Pn.to_parquet(DP / "sensor_daily_panel.parquet", index=False)
per = Pn.groupby("gid").agg(days=("y", "size"), first=("Date", "min"), last=("Date", "max"), grab_days=("y_grab", "count"))
print(f"{len(per)} sensor sites in the climate cube with >= {MIN_DAYS} days; {len(Pn):,} sensor days; "
      f"{int(per['grab_days'].sum()):,} same-day grab samples")
print(per.describe().round(1).to_string())
