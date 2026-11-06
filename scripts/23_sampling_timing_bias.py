"""Application: weather-timing bias of grab-sample nitrate statistics implied by the SK-DML response curves.

For each basin, the rain-driven log-nitrate anomaly on every day is a_t = sum_tau h(tau) * x~_{t-tau}, with
x~ = log1p(rain + melt) anomaly (basin x month demeaned) and h the basin's SK-DML response curve (0-180 d).
Bias of the ACTUAL sampling calendar = mean(a_t on sampled days) - mean(a_t on all days) (same years);
also reported for two hypothetical rules: 'dry-day' sampling (no input > 2 mm in previous 3 days) and monthly
fixed-day sampling. Values are converted to % concentration bias (exp(bias) - 1).
Usage: python 23_sampling_timing_bias.py <climate cache dir> [tag=real_all]
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from common import RES, load

CACHE = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "real_all"
src =(RES.parent / "scripts" / "21_rain_history_bins.py").read_text(encoding="utf-8")
ns = {}
exec(compile(src.split("obs, st, by = load()")[0].replace("CACHE = sys.argv[1]", "").replace("@njit(cache=True)", "@njit"), "21_head", "exec"), ns)
liquid_input = ns["liquid_input"]

obs, st, by = load()
kd = pd.read_csv(RES / f"rain_skdml_basin_params_{TAG}.csv")
H = np.load(RES / f"rain_skdml_response_curves_{TAG}.npy")
gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
dates = pd.date_range("1980-01-01", "2023-12-31", freq="D")
row = {g: i for i, g in enumerate(gids)}
months = dates.month.to_numpy()
sampled = obs.groupby("gid")["Date"].apply(lambda s: dates.get_indexer(s))
out = []
for k, g in enumerate(kd["gid"]):
    i = row[g]
    p = np.nan_to_num(np.asarray(cube[i:i + 1, :, 0], dtype=np.float64))
    t = np.nan_to_num(np.asarray(cube[i:i + 1, :, 4], dtype=np.float64))
    w = liquid_input(p, t)[0]
    x = np.log1p(w)
    for mth in range(1, 13):
        mm = months == mth
        x[mm] -= x[mm].mean()
    a = fftconvolve(x, H[k], mode="full")[: len(x)]
    idx = sampled.loc[g]
    idx = idx[(idx >= 180) & (idx < len(x))]
    if len(idx) < 30:
        continue
    yrs = dates.year.to_numpy()
    span = (yrs >= yrs[idx].min()) & (yrs <= yrs[idx].max())
    span[:180] = False
    allmean = a[span].mean()
    dry = span.copy()
    wet3 = np.convolve(w > 2.0, np.ones(3), mode="full")[: len(w)] > 0
    dry &= ~np.r_[False, wet3[:-1]]
    fixed = span & (dates.day.to_numpy() == 15)
    out.append({"gid": g, "n_samples": len(idx), "bias_actual_pct": 100 * (np.exp(a[idx].mean() - allmean) - 1),
                "bias_dryday_pct": 100 * (np.exp(a[dry].mean() - allmean) - 1),
                "bias_fixedday_pct": 100 * (np.exp(a[fixed].mean() - allmean) - 1),
                "sd_rain_driven_pct": 100 * (np.exp(a[span].std()) - 1),
                "p95_rain_driven_pct": 100 * (np.exp(np.quantile(a[span], 0.95)) - 1)})
R = pd.DataFrame(out).merge(kd[["gid", "net_response", "beta_slow", "peak_day"]], on="gid")
R.to_csv(RES / f"sampling_timing_bias_{TAG}.csv", index=False)
print(R.describe().round(2).to_string())
attr = st.set_index("gid")
for name, mask in {"tile>10%": attr.loc[R["gid"], "TOT_TILES_Early90s"].to_numpy() > 10,
                   "ag<10%": attr.loc[R["gid"], "TOT_AGRICULTURE_2011"].to_numpy() < 10}.items():
    print(name, R[mask][["bias_actual_pct", "bias_dryday_pct", "sd_rain_driven_pct", "p95_rain_driven_pct"]].median().round(2).to_dict())
