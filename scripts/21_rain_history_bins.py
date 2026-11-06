"""Daily liquid-water-input history (rain + degree-day snowmelt, NLDAS) for every nitrate sample.

For each sample at day t, the treatment history is x_tau = log1p(W_{t-tau}); it is summarised as bin means over
log-spaced lag bins (0..180 d) plus a placebo bin of FUTURE input (t+1..t+30). Output: data_processed/rain_bins.parquet.
Usage: python 21_rain_history_bins.py <climate cache dir>
"""
import sys

import numpy as np
import pandas as pd
from numba import njit

from common import DP, load

BINS = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 60), (61, 90), (91, 135), (136, 180)]
FUTURE = (-30, -1)
CACHE = sys.argv[1]


@njit(cache=True)
def liquid_input(p, t, ddf=3.0):
    out = np.zeros_like(p)
    for b in range(p.shape[0]):
        s = 0.0
        for i in range(p.shape[1]):
            rain = p[b, i] if t[b, i] > 0.0 else 0.0
            if t[b, i] <= 0.0:
                s += p[b, i]
            m = 0.0
            if t[b, i] > 0.0 and s > 0.0:
                m = min(s, ddf * t[b, i])
                s -= m
            out[b, i] = rain + m
    return out


obs, st, by = load()
gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
dates = pd.date_range("1980-01-01", "2023-12-31", freq="D")
P = np.nan_to_num(np.asarray(cube[:, :, 0], dtype=np.float64))
T = np.nan_to_num(np.asarray(cube[:, :, 4], dtype=np.float64))
X = np.log1p(liquid_input(P, T))
del P, T
cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, 1)], 1)
NT = X.shape[1]
bi = obs["gid"].map({g: i for i, g in enumerate(gids)}).to_numpy()
ti = dates.get_indexer(obs["Date"])
out = {"gid": obs["gid"].to_numpy(), "Date": obs["Date"].to_numpy(), "y": obs["y"].to_numpy(), "year": obs["year"].to_numpy()}
for name, (a, b) in [(f"R{a}_{b}", ab) for ab in BINS for a, b in [ab]] + [("F1_30", FUTURE)]:
    lo, hi = ti - b, ti - a + 1  # inclusive lag window [t-b, t-a]
    ok = (lo >= 0) & (hi <= NT)
    v = np.full(len(ti), np.nan, np.float32)
    v[ok] = (cs[bi[ok], hi[ok]] - cs[bi[ok], lo[ok]]) / (hi[ok] - lo[ok])
    out[name] = v
for a, b in BINS:  # mirrored FUTURE bins (placebo history): window [t+1+a, t+1+b]
    lo, hi = ti + 1 + a, ti + 2 + b
    ok = hi <= NT
    v = np.full(len(ti), np.nan, np.float32)
    v[ok] = (cs[bi[ok], hi[ok]] - cs[bi[ok], lo[ok]]) / (hi[ok] - lo[ok])
    out[f"FR{a}_{b}"] = v
for c in ["T7", "T30", "PET30", "SW7", "Q7", "SWE", "doy_s", "doy_c"]:
    out[c] = obs[c].to_numpy(np.float32)
df = pd.DataFrame(out)
df.to_parquet(DP / "rain_bins.parquet", index=False)
print(df.shape, df.isna().mean().round(4).to_dict())
