"""Empirical-Bayes SK-DML at the 12 nitrate-sensor basins: regional prior (attributes only; these basins excluded from
the kernel fit) vs direct (their own grab samples, regional timing) vs EB, each against the dense sensor reference
(36_sensor_validation.py). Reads rain_skdml_eb_basin_all.csv / rain_skdml_eb_sensor_curves_*.npy from 29 (task eb)."""
import numpy as np
import pandas as pd

from common import DP, RES

truth = pd.read_csv(RES / "sensor_validation_by_site.csv").set_index("gid")
eb = pd.read_csv(RES / "rain_skdml_eb_basin_all.csv")
eb = eb[eb["held_out"]]
hold = pd.read_csv(DP / "sensor_sites.csv")["gid"].tolist()
WIN = [(0, 3), (4, 10), (11, 30), (31, 90), (91, 180)]


def timing(h):
    net = h.sum()
    hd = np.clip(np.sign(net) * h, 0, None)
    C = np.cumsum(hd) / (hd.sum() + 1e-30)
    return {"net": net, "t50": int((C < 0.5).sum()), "peak": int(hd.argmax())}


rows = []
for pname, t in eb.groupby("prior"):
    t = t.set_index("gid").join(truth[["dense_net", "dense_net_se", "dense_sig"]], how="inner")
    mob = t["dense_net"] > 0
    for est in ("prior", "direct", "post"):
        col = f"{est}_net"
        for lab, msk in (("all sites", np.ones(len(t), bool)), ("dense |t|>2", t["dense_sig"].to_numpy())):
            v = t[msk].dropna(subset=[col])
            rows.append({"prior": pname, "estimator": {"prior": "regional prior", "direct": "direct (own grabs)", "post": "EB-SK-DML"}[est],
                         "sites": lab, "n": len(v), "spearman": v["dense_net"].corr(v[col], method="spearman"),
                         "sign_agree_pct": 100 * (np.sign(v["dense_net"]) == np.sign(v[col])).mean(),
                         "RMSE": np.sqrt(np.mean((v[col] - v["dense_net"]) ** 2)),
                         "mean_net_mobilising": v.loc[v["dense_net"] > 0, col].mean(),
                         "truth_mean_net_mobilising": v.loc[v["dense_net"] > 0, "dense_net"].mean()})
    print(f"[{pname}] own-data weight (slow) at sensor basins:", t["w_slow"].round(2).to_dict(), flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "eb_sensor_eval.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
dense_mob = [g for g in hold if g in truth.index and truth.loc[g, "dense_net"] > 0]
for pname in eb["prior"].unique():
    for lab in ("prior", "eb"):
        H = np.load(RES / f"rain_skdml_eb_sensor_curves_{pname}_{lab}_all.npy").astype(float)
        sel = [hold.index(g) for g in dense_mob]
        print(pname, lab, "mean curve over dense-mobilising sites:", timing(H[sel].mean(0)))
