"""Shared loading, feature definitions and metrics for IWAND-Nitrogen experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

EPS = 0.01  # mg-N/L offset for log transform
NVARS = ["DEP_N", "FERTAG_N", "FERTDEV_N", "FIX_N", "HUMAN_N", "LVST_N", "UPTAKE_N", "SURPLUS_N"]
CLIM = ["P1", "P3", "P7", "P14", "P30", "P90", "P180", "P365", "W7", "W30", "API90", "API97",
        "T1", "T7", "T30", "T90", "PET30", "PET90", "PET365", "WB30", "WB90", "WB365",
        "SWE", "MELT7", "MELT30", "SW7", "Q7"]
ID_COLS = ["gid", "Name", "State", "COMID", "Group", "STR_FTYPE", "Records", "Start", "End",
           "5th(mg-N/l)", "25th(mg-N/l)", "50th(mg-N/l)", "75th(mg-N/l)", "90th(mg-N/l)", "99th(mg-N/l)"]


def load(min_year=1980):
    obs = pd.read_parquet(DP / "obs_features.parquet")
    obs = obs[(obs["NO3"] >= 0) & obs["Date"].dt.year.ge(min_year)].copy()
    obs["year"] = obs["Date"].dt.year
    doy = obs["Date"].dt.dayofyear.to_numpy()
    obs["doy_s"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    obs["doy_c"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    obs["y"] = np.log(obs["NO3"].to_numpy() + EPS).astype(np.float32)
    st = pd.read_parquet(DP / "static.parquet")
    by = pd.read_parquet(DP / "basin_year.parquet")
    return obs, st, by


def static_features(st):
    num = st.drop(columns=[c for c in ID_COLS if c in st.columns]).select_dtypes("number")
    return num.columns.tolist()


def n_history(by, var="SURPLUS_N", max_lag=30, long=False):
    """Wide matrix of lagged annual inputs per gauge-year; pre-record years held at the first value.
    long=True (SURPLUS_N only) starts from the TREND-Nitrogen-spliced 1930 history instead of 1980."""
    piv = surplus_wide(by, long=True) if (long and var == "SURPLUS_N") else by.pivot(index="gid", columns="year", values=var)
    years = piv.columns.to_numpy()
    first = piv[years[0]]
    out = {}
    for lag in range(max_lag + 1):
        sh = piv.shift(lag, axis=1)
        sh = sh.apply(lambda col: col.fillna(first))
        out[lag] = sh
    return out, years


def surplus_wide(by, long=False):
    """Gauge x year SURPLUS_N. long=True uses the TREND-Nitrogen-spliced 1930-2023 series
    (13_build_long_surplus.py); otherwise the official IWAND 1980-2023 series."""
    if long:
        L = pd.read_parquet(DP / "basin_surplus_1930_2023_spliced.parquet")
        L.columns = [int(c) for c in L.columns]
        return L
    return by.pivot(index="gid", columns="year", values="SURPLUS_N")


def lag_window_features(by, var="SURPLUS_N", windows=((0, 0), (1, 5), (6, 10), (11, 20), (21, 30))):
    hist, years = n_history(by, var, max(w[1] for w in windows))
    rows = []
    for (a, b) in windows:
        m = sum(hist[l] for l in range(a, b + 1)) / (b - a + 1)
        rows.append(m.stack().rename(f"{var}_L{a}_{b}"))
    df = pd.concat(rows, axis=1).reset_index().rename(columns={"level_1": "year"})
    return df


def nse(o, s):
    o, s = np.asarray(o), np.asarray(s)
    d = np.sum((o - o.mean()) ** 2)
    return 1 - np.sum((o - s) ** 2) / d if d > 0 else np.nan


def kge(o, s):
    o, s = np.asarray(o), np.asarray(s)
    if o.std() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def trend_skill(df, pred_col="pred", y_col="y", min_years=12):
    """Across-basin agreement of decadal trends: Theil-Sen slope of annual mean log nitrate
    (observed vs predicted on the same sampled days). Returns Pearson r, slope-sign agreement, n basins."""
    from scipy.stats import theilslopes
    a = df.groupby(["gid", "year"]).agg(o=(y_col, "mean"), s=(pred_col, "mean"), n=(y_col, "size")).reset_index()
    a = a[a["n"] >= 4]
    rows = []
    for g, d in a.groupby("gid"):
        if d["year"].nunique() < min_years or d["year"].max() - d["year"].min() < 15:
            continue
        rows.append((theilslopes(d["o"], d["year"])[0], theilslopes(d["s"], d["year"])[0]))
    if len(rows) < 10:
        return {"trend_r": np.nan, "trend_sign": np.nan, "trend_basins": len(rows)}
    r = np.array(rows)
    big = np.abs(r[:, 0]) > 0.01
    return {"trend_r": float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]),
            "trend_sign": float(np.mean(np.sign(r[big, 0]) == np.sign(r[big, 1]))), "trend_basins": len(r)}


def evaluate(df, pred_col="pred", y_col="y", min_n=10):
    """Pooled metrics on log scale plus per-basin median NSE/KGE on log and linear scales."""
    o, s = df[y_col].to_numpy(), df[pred_col].to_numpy()
    out = {"n": len(df), "R2_log": 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2),
           "RMSE_log": float(np.sqrt(np.mean((o - s) ** 2)))}
    per = []
    for g, d in df.groupby("gid"):
        if len(d) < min_n:
            continue
        ol, sl = d[y_col].to_numpy(), d[pred_col].to_numpy()
        oc, sc = np.exp(ol) - EPS, np.exp(sl) - EPS
        per.append((nse(ol, sl), kge(oc, sc), nse(oc, sc), np.corrcoef(ol, sl)[0, 1] if ol.std() > 0 else np.nan))
    per = np.array(per)
    out.update({"basins": len(per), "medNSE_log": np.nanmedian(per[:, 0]), "medKGE": np.nanmedian(per[:, 1]),
                "medNSE": np.nanmedian(per[:, 2]), "medR_log": np.nanmedian(per[:, 3])})
    return out
