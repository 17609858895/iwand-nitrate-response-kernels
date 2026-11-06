"""Download USGS NWIS daily means (stat 00003) for the USGS-coded IWAND sites, 1979-2023.

Default: discharge (00060). Continuous in-situ nitrate sensors: params 99133,99137 (mg/L as N).
Raw tab-delimited (RDB) files go to data_external/<outdir>/<site>.rdb (untouched); a download log is written to
data_processed/<outdir>_download_log.csv. Re-running skips sites already downloaded.
Usage: python 31_download_nwis_dv.py [params=00060] [outdir=nwis_dv]
"""
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from common import DP, ROOT

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "00060"
NAME = sys.argv[2] if len(sys.argv) > 2 else "nwis_dv"
OUT = ROOT / "data_external" / NAME
OUT.mkdir(parents=True, exist_ok=True)
URL = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd=" + PARAMS + "&statCd=00003"
       "&startDT=1979-01-01&endDT=2023-12-31")
st = pd.read_parquet(DP / "static.parquet")
sites = sorted(st.loc[st["gid"].astype(str).str.startswith("USGS-"), "gid"].str[5:].unique())
rows = []
for i, site in enumerate(sites):
    f = OUT / f"{site}.rdb"
    status, nbytes = "cached", f.stat().st_size if f.exists() else 0
    if not f.exists():
        req = urllib.request.Request(URL.format(site=site), headers={"User-Agent": "python-urllib research-download"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                f.write_bytes(body)
                status, nbytes = f"http {r.status}", len(body)
                break
            except urllib.error.HTTPError as e:  # 404 = no discharge record for this site
                status = f"http {e.code}"
                if e.code == 404:
                    break
            except Exception as e:  # network hiccup: retry
                status = f"error {type(e).__name__}"
            time.sleep(5 * (attempt + 1))
        time.sleep(0.5)
    rows.append({"site": site, "status": status, "bytes": nbytes})
    if i % 25 == 0:
        print(i, site, status, nbytes, flush=True)
log = pd.DataFrame(rows)
log.to_csv(DP / f"{NAME}_download_log.csv", index=False)
print(log["status"].value_counts().to_string())
