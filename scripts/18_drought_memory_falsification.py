"""Falsification of the drought-memory interaction: future-deficit placebo, excluding 2012-2013, decade splits."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(__file__).with_name("17_drought_memory_profile.py").read_text(encoding="utf-8").split("rows = []")[0]
ns = {"__file__": str(Path(__file__).with_name("17_drought_memory_profile.py")), "__name__": "dm"}
sys.argv = [sys.argv[0], sys.argv[1]]
exec(compile(src, "17_head", "exec"), ns)
obs, fe_ols, window_deficit, z, attr, RES = ns["obs"], ns["fe_ols"], ns["window_deficit"], ns["z"], ns["attr"], ns["RES"]
cs, bi, ti = ns["cs"], ns["bi"], ns["ti"]


def future_deficit(start, length):  # window strictly after sampling (placebo)
    a = ti + start
    b = ti + start + length
    ok = b < cs.shape[1]
    v = np.full(len(ti), np.nan)
    v[ok] = -(cs[bi[ok], b[ok]] - cs[bi[ok], a[ok]])
    return v


def prep(dry):
    d = obs.copy()
    d["dry"] = dry
    d["dry"] = d.groupby(["gid", "m"])["dry"].transform(z)
    d["wxd"] = d["wet"] * d["dry"]
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=["wet", "dry", "wxd", "ya"])


rows = []
agmask = lambda d: d["gid"].map(attr["TOT_AGRICULTURE_2011"]).to_numpy() > 40
tests = {"past_90to455d": prep(window_deficit(90, 365)), "future_placebo_30to395d": prep(future_deficit(30, 365))}
for name, d in tests.items():
    for grp, mask in (("all", np.ones(len(d), bool)), ("ag>40%", agmask(d))):
        b, t, _ = fe_ols(d[mask], ["wet", "dry", "wxd"])
        rows.append({"test": name, "group": grp, "wxd_b": b[2], "wxd_t": t[2], "dry_b": b[1], "dry_t": t[1]})
d = tests["past_90to455d"]
for name, mask in {"excl_2012_2013": ~d["year"].isin([2012, 2013]), "1980-1999": d["year"] < 2000,
                   "2000-2011": d["year"].between(2000, 2011), "2012-2023": d["year"] >= 2012}.items():
    for grp, m2 in (("all", np.ones(len(d), bool)), ("ag>40%", agmask(d))):
        b, t, _ = fe_ols(d[mask.to_numpy() & m2], ["wet", "dry", "wxd"])
        rows.append({"test": name, "group": grp, "wxd_b": b[2], "wxd_t": t[2], "dry_b": b[1], "dry_t": t[1]})
R = pd.DataFrame(rows)
R.to_csv(RES / "drought_memory_falsification.csv", index=False)
print(R.round(4).to_string(index=False))
