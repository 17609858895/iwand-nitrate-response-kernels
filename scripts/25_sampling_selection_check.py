"""Sampling-selection diagnostic: are grab samples taken on systematically drier/wetter days than the basin climatology?
Compares rain bins on sampled days with the same basin-month climatology (all days), and reports weekday shares."""
import sys

import numpy as np
import pandas as pd

from common import DP, RES

d = pd.read_parquet(DP / "rain_bins.parquet")
dt = pd.DatetimeIndex(d["Date"])
wd = pd.Series(dt.dayofweek).value_counts(normalize=True).sort_index()
print("weekday share of samples (Mon=0):", wd.round(3).to_dict())
CACHE = sys.argv[1]
src = (RES.parent / "scripts" / "21_rain_history_bins.py").read_text(encoding="utf-8")
ns = {}
exec(compile(src.split("obs, st, by = load()")[0].replace("CACHE = sys.argv[1]", "").replace("@njit(cache=True)", "@njit"), "21_head", "exec"), ns)
gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
dates = pd.date_range("1980-01-01", "2023-12-31", freq="D")
P = np.nan_to_num(np.asarray(cube[:, :, 0], dtype=np.float64))
T = np.nan_to_num(np.asarray(cube[:, :, 4], dtype=np.float64))
X = np.log1p(ns["liquid_input"](P, T))
bi = d["gid"].map({g: i for i, g in enumerate(gids)}).to_numpy()
ti = dates.get_indexer(dt)
mon = dates.month.to_numpy()
# climatological mean of same-day and previous-3-day input for each basin-month
cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, 1)], 1)
prev3 = np.zeros_like(X)
prev3[:, 3:] = (cs[:, 3:-1] - cs[:, :-4]) / 3
rows = []
for lagname, arr in [("same_day", X), ("prev_3d", prev3)]:
    clim = np.zeros((X.shape[0], 13))
    for m in range(1, 13):
        clim[:, m] = arr[:, mon == m].mean(1)
    samp = arr[bi, ti]
    ref = clim[bi, dt.month.to_numpy()]
    diff = samp - ref
    rows.append({"window": lagname, "mean_diff_log1p_mm": float(diff.mean()), "median_basin_diff": float(pd.Series(diff).groupby(bi).mean().median()),
                 "share_basins_drier": float((pd.Series(diff).groupby(bi).mean() < 0).mean())})
R = pd.DataFrame(rows)
R.to_csv(RES / "sampling_selection_check.csv", index=False)
print(R.round(4).to_string(index=False))
