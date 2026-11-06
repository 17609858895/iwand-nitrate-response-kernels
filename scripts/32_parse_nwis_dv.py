"""Parse downloaded NWIS daily-mean RDB files into data_processed/<out>.parquet (gid, Date, <col>, qual).

Where a site reports several daily-mean time series (any of the requested parameter codes), the one with most valid
values is kept. Non-numeric entries (Ice, Eqp, Ssn, ...) become missing; discharge must be > 0, other parameters >= 0.
Usage (isolated mode, downloaded files are untrusted data):
  python -I 32_parse_nwis_dv.py <raw dir> [params=00060] [out=nwis_dv] [col=Q_cfs]
  python -I 32_parse_nwis_dv.py <raw dir> 99133,99137 nwis_sensor_no3 NO3_mgL
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DP  # noqa: E402

RAW = Path(sys.argv[1])
PARAMS = (sys.argv[2] if len(sys.argv) > 2 else "00060").split(",")
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "nwis_dv"
COL = sys.argv[4] if len(sys.argv) > 4 else "Q_cfs"
frames, rows = [], []
for f in sorted(RAW.glob("*.rdb")):
    lines = [ln for ln in f.read_text(encoding="utf-8", errors="replace").splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3 or not lines[0].startswith("agency_cd"):
        rows.append({"site": f.stem, "n_days": 0})
        continue
    hdr = lines[0].split("\t")
    vcols = [c for c in hdr if any(c.endswith(f"_{pc}_00003") for pc in PARAMS)]
    if not vcols:  # header only, no daily-mean discharge series
        rows.append({"site": f.stem, "n_days": 0})
        continue
    t = pd.DataFrame([ln.split("\t") for ln in lines[2:]], columns=hdr)
    best = max(vcols, key=lambda c: pd.to_numeric(t[c], errors="coerce").notna().sum())
    q = pd.to_numeric(t[best], errors="coerce")
    ok = q > 0 if PARAMS == ["00060"] else q >= 0
    out = pd.DataFrame({"gid": "USGS-" + f.stem, "Date": pd.to_datetime(t["datetime"]), COL: q.where(ok),
                        "qual": t[best + "_cd"].astype(str)})
    out = out.dropna(subset=[COL])
    frames.append(out)
    rows.append({"site": f.stem, "n_days": len(out), "n_series": len(vcols), "first": out["Date"].min(), "last": out["Date"].max(),
                 "pct_provisional": 100 * out["qual"].str.contains("P").mean() if len(out) else np.nan})
Q = pd.concat(frames, ignore_index=True)
Q.to_parquet(DP / f"{OUTNAME}.parquet", index=False)
S = pd.DataFrame(rows)
S.to_csv(DP / f"{OUTNAME}_site_summary.csv", index=False)
print(f"{(S['n_days'] > 0).sum()} of {len(S)} sites with data; {len(Q):,} daily values")
print(S.loc[S["n_days"] > 0, ["n_days", "n_series", "pct_provisional"]].describe().round(1).to_string())
