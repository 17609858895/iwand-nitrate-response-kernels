"""Build modelling tables from the official IWAND-Nitrogen CSV zip (read-only).

Outputs (data_processed/):
  obs_features.parquet   one row per gauge-day nitrate record with antecedent climate features
  basin_year.parquet     annual climate aggregates + 8 N forcings per gauge-year
  static.parquet         basin attributes + nitrate summary
  climate_cube.npy       float32 [gauge, day, 11] daily forcings (scratch cache, optional)
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "IWAND-Nitrogen-CSV.zip"
OUT = ROOT / "data_processed"
OUT.mkdir(exist_ok=True)
PFX = "IWAND-Nitrogen-CSV/"
CLIM_VARS = ["APCP", "DLWRF", "DSWRF", "PRES", "TMP", "UGRD", "CONVfrac", "VGRD", "SPFH", "PEVAP", "CAPE"]
DATES = pd.date_range("1980-01-01", "2023-12-31", freq="D")


@njit(cache=True)
def snow_model(p, t, ddf=3.0):
    """Degree-day snow bucket: returns SWE and melt (mm/day)."""
    n = p.shape[0]
    swe = np.zeros(n)
    melt = np.zeros(n)
    s = 0.0
    for i in range(n):
        if t[i] <= 0.0:
            s += p[i]
        m = 0.0
        if t[i] > 0.0 and s > 0.0:
            m = min(s, ddf * t[i])
            s -= m
        swe[i] = s
        melt[i] = m
    return swe, melt


@njit(cache=True)
def api(x, k):
    """Antecedent index with daily decay k."""
    out = np.zeros(x.shape[0])
    a = 0.0
    for i in range(x.shape[0]):
        a = a * k + x[i]
        out[i] = a
    return out


def trailing(x, w, how="sum"):
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    s = np.full(x.shape[0], np.nan)
    s[w - 1:] = c[w:] - c[:-w]
    return s if how == "sum" else s / w


def process_gauge(gid, obs_dates):
    with zipfile.ZipFile(ZIP) as z:
        c = pd.read_csv(z.open(f"{PFX}climate_forcings/Climate_{gid}.csv"), parse_dates=["Date"])
    c = c.set_index("Date").reindex(DATES)
    arr = c[CLIM_VARS].to_numpy(np.float64)
    p, t, pet = arr[:, 0], arr[:, 4], arr[:, 9]
    sw, spfh = arr[:, 2], arr[:, 8]
    rain = np.where(t > 0, p, 0.0)
    swe, melt = snow_model(np.nan_to_num(p), np.nan_to_num(t))
    winp = rain + melt  # liquid water input
    feats = {
        "P1": p, "P3": trailing(p, 3), "P7": trailing(p, 7), "P14": trailing(p, 14),
        "P30": trailing(p, 30), "P90": trailing(p, 90), "P180": trailing(p, 180), "P365": trailing(p, 365),
        "W7": trailing(winp, 7), "W30": trailing(winp, 30),
        "API90": api(np.nan_to_num(winp), 0.90), "API97": api(np.nan_to_num(winp), 0.97),
        "T1": t, "T7": trailing(t, 7, "mean"), "T30": trailing(t, 30, "mean"), "T90": trailing(t, 90, "mean"),
        "PET30": trailing(pet, 30), "PET90": trailing(pet, 90), "PET365": trailing(pet, 365),
        "WB30": trailing(winp - pet, 30), "WB90": trailing(winp - pet, 90), "WB365": trailing(p - pet, 365),
        "SWE": swe, "MELT7": trailing(melt, 7), "MELT30": trailing(melt, 30),
        "SW7": trailing(sw, 7, "mean"), "Q7": trailing(spfh, 7, "mean"),
    }
    idx = DATES.get_indexer(pd.DatetimeIndex(obs_dates))
    ok = idx >= 0
    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32)[idx[ok]] for k, v in feats.items()})
    df.insert(0, "Date", pd.DatetimeIndex(obs_dates)[ok])
    df.insert(0, "gid", gid)
    # annual aggregates
    yr = DATES.year
    ann = pd.DataFrame({"year": yr, "P": p, "T": t, "PET": pet, "MELT": melt, "W": winp}).groupby("year").agg(
        P=("P", "sum"), T=("T", "mean"), PET=("PET", "sum"), MELT=("MELT", "sum"), W=("W", "sum"))
    ann["P_PET"] = ann["P"] - ann["PET"]
    ann = ann.reset_index()
    ann.insert(0, "gid", gid)
    return df, ann, arr.astype(np.float32)


def main():
    z = zipfile.ZipFile(ZIP)
    obs = pd.read_csv(z.open(f"{PFX}in_situ_data/WQP_US_NO3.csv"), parse_dates=["Date"])
    obs = obs.rename(columns={"Gauge-ID": "gid", "Nitrate(mg-N/l)": "NO3"})
    att = pd.read_csv(z.open(f"{PFX}attributes/Basin_Attributes.csv")).rename(columns={"Gauge-ID": "gid"})
    summ = pd.read_csv(z.open(f"{PFX}in_situ_data/WQP_US_NO3_Summary.csv")).rename(columns={"Gauge-ID": "gid"})
    gids = att["gid"].tolist()
    # N forcings
    nlist = []
    for g in gids:
        n = pd.read_csv(z.open(f"{PFX}nitrogen_forcings/Nitrogen_{g}.csv"), parse_dates=["Date"])
        n["year"] = n["Date"].dt.year
        n["gid"] = g
        nlist.append(n.drop(columns="Date"))
    nf = pd.concat(nlist, ignore_index=True)
    z.close()

    groups = {g: d["Date"].values for g, d in obs.groupby("gid")}
    res = Parallel(n_jobs=24, verbose=5)(delayed(process_gauge)(g, groups[g]) for g in gids)
    feat = pd.concat([r[0] for r in res], ignore_index=True)
    ann = pd.concat([r[1] for r in res], ignore_index=True)
    cube = np.stack([r[2] for r in res])

    o = obs.merge(feat, on=["gid", "Date"], how="left")
    o.to_parquet(OUT / "obs_features.parquet", index=False)
    ann.merge(nf, on=["gid", "year"], how="left").to_parquet(OUT / "basin_year.parquet", index=False)
    att.merge(summ.drop(columns=["State", "Lat", "Lon"]), on="gid").to_parquet(OUT / "static.parquet", index=False)
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    np.save(cache / "climate_cube.npy", cube)
    pd.Series(gids).to_csv(cache / "climate_cube_gids.csv", index=False)
    print(o.shape, ann.shape, cube.shape, o.isna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
