"""How long is the 'drought memory' of river nitrate? Profile the wet x prior-deficit interaction over the
length/offset of the antecedent water-balance window (daily NLDAS P - PEVAP from the climate cube).
Treatment windows end 90 days before sampling (so they do not overlap the current-wetness window)."""
import sys

import numpy as np
import pandas as pd

from common import RES, load

CACHE = sys.argv[1]
obs, st, by = load()
gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
dates = pd.date_range("1980-01-01", "2023-12-31", freq="D")
wb = np.nan_to_num(np.asarray(cube[:, :, 0], dtype=np.float64) - np.asarray(cube[:, :, 9], dtype=np.float64))
cs = np.concatenate([np.zeros((wb.shape[0], 1)), np.cumsum(wb, 1)], 1)
bi = obs["gid"].map({g: i for i, g in enumerate(gids)}).to_numpy()
ti = dates.get_indexer(obs["Date"])
obs["m"] = obs["Date"].dt.month
obs["ya"] = obs["y"] - obs.groupby(["gid", "m"])["y"].transform("mean")
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
obs["wet"] = obs.groupby(["gid", "m"])["P90"].transform(z)
attr = st.set_index("gid")
ag = obs["gid"].map(attr["TOT_AGRICULTURE_2011"]).to_numpy()


def window_deficit(offset, length):
    a = np.clip(ti - offset - length, 0, None)
    b = np.clip(ti - offset, 0, None)
    v = cs[bi, b] - cs[bi, a]
    v[(ti - offset - length) < 0] = np.nan
    return -v  # positive = drier


def fe_ols(d, cols):
    D = d[["ya"] + cols].to_numpy(float).copy()
    g1, g2 = pd.factorize(d["gid"])[0], pd.factorize(d["year"])[0]
    for _ in range(10):
        for g in (g1, g2):
            s = np.zeros((g.max() + 1, D.shape[1]))
            np.add.at(s, g, D)
            D -= (s / np.bincount(g)[:, None])[g]
    y, X = D[:, 0], D[:, 1:]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    U = np.zeros((g1.max() + 1, X.shape[1]))
    np.add.at(U, g1, X * e[:, None])
    se = np.sqrt(np.diag(XtXi @ (U.T @ U) @ XtXi))
    return b, b / se, float(e @ e)


rows = []
for offset in (90, 270, 455, 820):
    for length in (90, 180, 365, 730):
        obs["dry"] = window_deficit(offset, length)
        obs["dry"] = obs.groupby(["gid", "m"])["dry"].transform(z)
        obs["wxd"] = obs["wet"] * obs["dry"]
        d0 = obs.replace([np.inf, -np.inf], np.nan).dropna(subset=["wet", "dry", "wxd", "ya"])
        for grp, mask in (("all", np.ones(len(d0), bool)), ("ag>40%", d0["gid"].map(attr["TOT_AGRICULTURE_2011"]).to_numpy() > 40)):
            b, t, sse = fe_ols(d0[mask], ["wet", "dry", "wxd"])
            rows.append({"offset_days": offset, "length_days": length, "group": grp, "n": int(mask.sum()),
                         "wet_b": b[0], "dry_b": b[1], "wxd_b": b[2], "wet_t": t[0], "dry_t": t[1], "wxd_t": t[2]})
        print(offset, length, " | ".join(f"{r['group']}: dry {r['dry_b']:+.3f}(t{r['dry_t']:.1f}) wxd {r['wxd_b']:+.3f}(t{r['wxd_t']:.1f})"
                                         for r in rows[-2:]), flush=True)
pd.DataFrame(rows).to_csv(RES / "drought_memory_profile.csv", index=False)
