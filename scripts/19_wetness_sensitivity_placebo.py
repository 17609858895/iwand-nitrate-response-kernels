"""Is the contemporaneous wetness effect (mobilisation vs dilution) robust? Placebo = precipitation in the 90 days AFTER sampling."""
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
ok = ti + 91 < cp.shape[1]
fut = np.full(len(ti), np.nan)
fut[ok] = cp[bi[ok], ti[ok] + 91] - cp[bi[ok], ti[ok] + 1]
obs["wet_future"] = fut
obs["wet_future"] = obs.groupby(["gid", "m"])["wet_future"].transform(z)
d = obs.replace([np.inf, -np.inf], np.nan).dropna(subset=["wet", "wet_future", "ya"])
ag = d["gid"].map(attr["TOT_AGRICULTURE_2011"]).to_numpy()
tile = d["gid"].map(attr["TOT_TILES_Early90s"]).to_numpy()
rows = []
for grp, mask in {"all": np.ones(len(d), bool), "ag>40%": ag > 40, "ag<10%": ag < 10, "tile>10%": tile > 10}.items():
    b, t, _ = fe_ols(d[mask], ["wet", "wet_future"])
    rows.append({"group": grp, "wet_b": b[0], "wet_t": t[0], "future_wet_b": b[1], "future_wet_t": t[1]})
    b2, t2, _ = fe_ols(d[mask & (d["year"] < 2012).to_numpy()], ["wet", "wet_future"])
    rows.append({"group": grp + " (<2012)", "wet_b": b2[0], "wet_t": t2[0], "future_wet_b": b2[1], "future_wet_t": t2[1]})
R = pd.DataFrame(rows)
R.to_csv(RES / "wetness_sensitivity_placebo.csv", index=False)
print(R.round(4).to_string(index=False))
