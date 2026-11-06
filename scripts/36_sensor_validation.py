"""Reference comparison with USGS continuous nitrate sensors (daily means) at IWAND basins.

Reference: per-site distributed-lag (DL) response estimated from the DENSE daily sensor series (site x month + year FE,
rain bins residualised on weather controls, year-month clustered SE). Bin coefficients c_b give a daily kernel
h_tau = c_b / width_b, so sum(h) = net response, comparable with the SK-DML curves.
Compared against
  1. SK-DML kernels predicted from attributes alone for the held-out sensor basins (29_rain_skdml_checks.py holdout:
     their grab samples were excluded from the fit);
  2. per-site DL estimated from grab-like subsamples of the same sensor series (12 / yr on Mon-Wed, R draws);
  3. per-site DL estimated from the site's actual IWAND grab samples (all years).
Plus pooled recovery of the mean profile and same-day grab vs sensor agreement.
Usage: python 36_sensor_validation.py
"""
import numpy as np
import pandas as pd

from common import DP, RES

BINS = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 60), (61, 90), (91, 135), (136, 180)]
RC = [f"R{a}_{b}" for a, b in BINS]
WIDTH = np.array([b - a + 1 for a, b in BINS], float)
CTRL = ["T7", "T30", "PET30", "SW7", "Q7", "SWE"]
WIN = [(0, 3), (4, 10), (11, 30), (31, 90), (91, 180)]
NDRAW = 100
rng = np.random.default_rng(0)


def within(D, groups, n_iter=10):
    D = D.copy()
    for _ in range(n_iter):
        for g in groups:
            s = np.zeros((g.max() + 1, D.shape[1]))
            np.add.at(s, g, D)
            D -= (s / np.bincount(g)[:, None])[g]
    return D


def dl_fit(df, site_fe=False):
    """DL of y on 12 rain bins (+ future bin); month(-by-site) + year FE; returns coef, cluster SE, net, net SE."""
    df = df.dropna(subset=["y"] + RC + ["F1_30"] + CTRL)
    m = pd.DatetimeIndex(df["Date"]).month.to_numpy()
    key = (df["gid"].astype(str) + "_" + pd.Series(m, index=df.index).astype(str)) if site_fe else pd.Series(m, index=df.index)
    G = [pd.factorize(key)[0], pd.factorize(df["year"])[0]]
    D = within(df[["y"] + RC + ["F1_30"] + CTRL].to_numpy(float), G)
    y, X, C = D[:, 0], D[:, 1:14], D[:, 14:]
    X = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    cl = pd.factorize(df["gid"].astype(str) + pd.DatetimeIndex(df["Date"]).strftime("_%Y%m"))[0]
    XtXi = np.linalg.pinv(X.T @ X)
    U = np.zeros((cl.max() + 1, X.shape[1]))
    np.add.at(U, cl, X * e[:, None])
    V = XtXi @ (U.T @ U) @ XtXi
    w = np.r_[np.ones(12), 0.0]
    return b[:12], np.sqrt(np.diag(V))[:12], float(w @ b), float(np.sqrt(w @ V @ w)), len(df)


def curve(c):
    return np.repeat(c / WIDTH, WIDTH.astype(int))[:181]


def timing(h):
    net = h.sum()
    hd = np.clip(np.sign(net) * h, 0, None)
    C = np.cumsum(hd) / (hd.sum() + 1e-30)
    r = {"net": net, "t50": int((C < 0.5).sum()), "peak": int(hd.argmax())}
    r.update({f"share{a}_{b}": hd[a:b + 1].sum() / (hd.sum() + 1e-30) for a, b in WIN})
    return r


P = pd.read_parquet(DP / "sensor_daily_panel.parquet")
grab = pd.read_parquet(DP / "rain_bins.parquet", columns=["gid", "Date", "y", "year"] + RC + ["F1_30"] + CTRL)
st = pd.read_parquet(DP / "static.parquet").set_index("gid")
hold = pd.read_csv(RES / "rain_skdml_holdout_all.csv")
Hc = {m: np.load(RES / f"rain_skdml_holdout_curves_{m}_all.npy") for m in hold["model"].unique()}
hold_gids = hold.loc[hold["model"] == "SK-DML_K2_regional", "gid"].tolist()

rows, curves_dense = [], {}
for g, df in P.groupby("gid"):
    c, se, net, net_se, n = dl_fit(df)
    h = curve(c)
    curves_dense[g] = h
    r = {"gid": g, "n_days": n, "years": df["year"].nunique(), "dense_net": net, "dense_net_se": net_se,
         **{f"dense_{k}": v for k, v in timing(h).items() if k != "net"}}
    # grab-like subsamples of the sensor series: 12 per year on Mon-Wed
    wk = pd.DatetimeIndex(df["Date"]).dayofweek.to_numpy() <= 2
    sub = []
    for _ in range(NDRAW):
        idx = df[wk].groupby("year", group_keys=False).apply(lambda x: x.sample(min(12, len(x)), random_state=int(rng.integers(1e9))))
        try:
            sub.append(dl_fit(idx)[2])
        except np.linalg.LinAlgError:
            continue
    r["sub12_net_median"], r["sub12_net_sd"] = np.median(sub), np.std(sub)
    r["sub12_sign_agree"] = np.mean(np.sign(sub) == np.sign(net))
    gb = grab[grab["gid"] == g]
    if len(gb) >= 60:
        cg = dl_fit(gb)
        r["grab_net"], r["grab_net_se"], r["grab_n"] = cg[2], cg[3], cg[4]
    for m, H in Hc.items():
        if g in hold_gids:
            th = timing(H[hold_gids.index(g)].astype(float))
            r.update({f"{m}|{k}": v for k, v in th.items()})
    rows.append(r)
V = pd.DataFrame(rows).set_index("gid")
V = V.join(st[["TOT_TILES_Early90s", "TOT_AGRICULTURE_2011", "TOT_BFI", "WSHDBDSqKM"]])
V["dense_sig"] = (V["dense_net"].abs() / V["dense_net_se"]) > 2
V.to_csv(RES / "sensor_validation_by_site.csv")

pd.set_option("display.width", 200)
print(f"{len(V)} sensor sites; median {V['n_days'].median():.0f} days, {V['years'].median():.0f} years; "
      f"{V['dense_sig'].sum()} with |t(net)| > 2")
summ = []
for est, col in [("SK-DML linear, held out (attributes only)", "SK-DML_K2_regional|net"),
                 ("SK-DML MLP, held out (attributes only)", "SK-DML_K2_regional_mlp|net"),
                 ("per-site DL, 12/yr sensor subsample (median of draws)", "sub12_net_median"),
                 ("per-site DL, actual IWAND grab samples", "grab_net")]:
    if col not in V:
        continue
    for lab, msk in [("all sites", np.ones(len(V), bool)), ("dense |t|>2", V["dense_sig"].to_numpy())]:
        v = V[msk].dropna(subset=[col])
        summ.append({"estimator": est, "sites": lab, "n": len(v), "spearman_net": v["dense_net"].corr(v[col], method="spearman"),
                     "pearson_net": v["dense_net"].corr(v[col]), "sign_agree_pct": 100 * (np.sign(v["dense_net"]) == np.sign(v[col])).mean()})
S = pd.DataFrame(summ)
S.to_csv(RES / "sensor_validation_summary.csv", index=False)
print(S.round(3).to_string(index=False))

# pooled timing: mean dense curve vs mean predicted curve over sites mobilising in the dense truth
mob = V.index[V["dense_net"] > 0].tolist()
rows = [{"curve": "dense sensor (mobilising sites)", **timing(np.mean([curves_dense[g] for g in mob], 0))}]
for m, H in Hc.items():
    sel = [hold_gids.index(g) for g in mob if g in hold_gids]
    rows.append({"curve": f"{m} held out (same sites)", **timing(H[sel].astype(float).mean(0))})
# pooled DL on all sensor days vs on 12/yr subsamples (site x month + year FE)
c, se, net, _, _ = dl_fit(P, site_fe=True)
rows.append({"curve": "pooled DL, dense", **timing(curve(c))})
wk = pd.DatetimeIndex(P["Date"]).dayofweek.to_numpy() <= 2
pooled = [dl_fit(P[wk].groupby(["gid", "year"], group_keys=False).apply(lambda x: x.sample(min(12, len(x)), random_state=k)), site_fe=True)[0]
          for k in range(20)]
rows.append({"curve": "pooled DL, 12/yr subsample (mean of 20)", **timing(curve(np.mean(pooled, 0)))})
T = pd.DataFrame(rows)
T.to_csv(RES / "sensor_validation_pooled_timing.csv", index=False)
print(T.round(3).to_string(index=False))
prof = pd.DataFrame({"bin": RC, "dense": c, "dense_se": se, "sub12_mean": np.mean(pooled, 0), "sub12_sd": np.std(pooled, 0)})
prof.to_csv(RES / "sensor_validation_pooled_profile.csv", index=False)
print(prof.round(4).to_string(index=False))

# same-day grab vs sensor
gd = P.dropna(subset=["y_grab"])
print(f"same-day grab vs sensor: n={len(gd)}, sites={gd['gid'].nunique()}, r={np.corrcoef(gd['y'], gd['y_grab'])[0, 1]:.3f}, "
      f"median log ratio grab-sensor={np.median(gd['y_grab'] - gd['y']):+.3f}")
