"""Distributed-lag response of nitrate to daily precipitation (exogenous): lag bins 0-3, 4-10, 11-30, 31-90, 91-180 d,
plus future placebo bin (+1..+30 d); basin x month and year FE; by land-use / BFI class."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(__file__).with_name("17_drought_memory_profile.py").read_text(encoding="utf-8").split("rows = []")[0]
ns = {"__file__": str(Path(__file__).with_name("17_drought_memory_profile.py")), "__name__": "dm"}
sys.argv = [sys.argv[0], sys.argv[1]]
exec(compile(src, "17_head", "exec"), ns)
obs, fe_ols, z, attr, RES, cube, bi, ti = (ns[k] for k in ("obs", "fe_ols", "z", "attr", "RES", "cube", "bi", "ti"))
P = np.nan_to_num(np.asarray(cube[:, :, 0], dtype=np.float64))
cp = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, 1)], 1)
NT = P.shape[1]
BINS = {"L0_3": (0, 3), "L4_10": (4, 10), "L11_30": (11, 30), "L31_90": (31, 90), "L91_180": (91, 180), "F1_30": (-30, -1)}
for name, (a, b) in BINS.items():
    lo, hi = ti - b, ti - a + 1  # inclusive window [t-b, t-a]
    ok = (lo >= 0) & (hi <= NT)
    v = np.full(len(ti), np.nan)
    v[ok] = cp[bi[ok], hi[ok]] - cp[bi[ok], lo[ok]]
    obs[name] = v
    obs[name] = obs.groupby(["gid", "m"])[name].transform(z)
cols = list(BINS)
d = obs.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["ya"])
A = d["gid"].map(attr["TOT_AGRICULTURE_2011"]).to_numpy()
T = d["gid"].map(attr["TOT_TILES_Early90s"]).to_numpy()
B = d["gid"].map(attr["TOT_BFI"]).to_numpy()
q = np.nanquantile(attr["TOT_BFI"], [1 / 3, 2 / 3])
rows = []
for grp, mask in {"all": np.ones(len(d), bool), "tile>10%": T > 10, "ag>40%": A > 40, "ag<10%": A < 10,
                  "BFI_low": B < q[0], "BFI_high": B >= q[1]}.items():
    b, t, _ = fe_ols(d[mask], cols)
    rows.append({"group": grp, **{f"{c}_b": v for c, v in zip(cols, b)}, **{f"{c}_t": v for c, v in zip(cols, t)}})
    print(f"{grp:9s}", " ".join(f"{c}:{bb:+.3f}({tt:.1f})" for c, bb, tt in zip(cols, b, t)), flush=True)
pd.DataFrame(rows).to_csv(RES / "rain_response_lag_bins.csv", index=False)
