"""Entity-aware LSTM baseline (CAMELS-style) for sparse daily nitrate.

Inputs per day: 11 NLDAS forcings + day-of-year + 8 annual N inputs + static attributes.
Loss only on observed days inside the final `PRED` days of each window (warm-up = SEQ - PRED).
Usage: python 03_lstm_baseline.py temporal|block0..block4 [cache_dir]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import NVARS, RES, evaluate, load, static_features

torch.set_num_threads(12)
torch.manual_seed(0)
np.random.seed(0)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "temporal"
CACHE = sys.argv[2] if len(sys.argv) > 2 else str(RES.parent / "data_processed")
SEQ, PRED, HID, BATCH, EPOCHS, PER_EPOCH = 365, 90, 96, 256, 18, 30000

obs, st, by = load()
gids = pd.read_csv(f"{CACHE}/climate_cube_gids.csv").iloc[:, 0].tolist()
gidx = {g: i for i, g in enumerate(gids)}
cube = np.load(f"{CACHE}/climate_cube.npy", mmap_mode="r")
dates = pd.date_range("1980-01-01", "2023-12-31", freq="D")
NB, NT, NC = cube.shape

SF = static_features(st)
S = st.set_index("gid").loc[gids, SF].to_numpy(np.float64)
pos = np.nanmin(S, 0) >= 0
S = np.where(pos[None, :], np.log1p(np.clip(S, 0, None)), S)  # log-compress non-negative attributes
S = np.nan_to_num((S - np.nanmean(S, 0)) / (np.nanstd(S, 0) + 1e-6)).astype(np.float32)

# annual N inputs broadcast to days
Nmat = by.pivot_table(index="gid", columns="year", values=NVARS).loc[gids]
yrs = dates.year.to_numpy() - 1980
Narr = np.stack([Nmat[v].to_numpy()[:, yrs] for v in NVARS], -1).astype(np.float32)  # [NB, NT, 8]
Narr = np.log1p(np.clip(Narr, 0, None))
nmu, nsd = Narr.mean((0, 1)), Narr.std((0, 1)) + 1e-6
Narr = (Narr - nmu) / nsd

C = np.array(cube, dtype=np.float32)
C[:, :, 0] = np.log1p(np.clip(C[:, :, 0], 0, None))  # precipitation
C[:, :, 9] = np.sign(C[:, :, 9]) * np.log1p(np.abs(C[:, :, 9]))  # PET
C[:, :, 10] = np.log1p(np.clip(C[:, :, 10], 0, None))  # CAPE
cmu, csd = np.nanmean(C, (0, 1)), np.nanstd(C, (0, 1)) + 1e-6
C = np.nan_to_num((C - cmu) / csd).astype(np.float32)
doy = dates.dayofyear.to_numpy()
D = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], -1).astype(np.float32)

Y = np.full((NB, NT), np.nan, np.float32)
bi = obs["gid"].map(gidx).to_numpy()
ti = dates.get_indexer(obs["Date"])
Y[bi, ti] = obs["y"].to_numpy()
ymu, ysd = np.nanmean(Y), np.nanstd(Y)
Yn = (Y - ymu) / ysd

# split masks
yearmat = dates.year.to_numpy()
if SPLIT == "temporal":
    train_basins = np.arange(NB)
    test_basins = np.arange(NB)
    train_tmask = yearmat <= 2012
    test_tmask = yearmat >= 2013
else:
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GroupKFold
    k = int(SPLIT.replace("block", ""))
    xy = st.set_index("gid").loc[gids, ["Lat", "Lon"]]
    blk = KMeans(60, n_init=5, random_state=0).fit_predict(xy)
    folds = list(GroupKFold(5).split(np.zeros(NB), groups=blk))
    train_basins, test_basins = folds[k]
    train_tmask = np.ones(NT, bool)
    test_tmask = np.ones(NT, bool)

Ytr = np.where(train_tmask[None, :], Yn, np.nan)
Ytr[np.setdiff1d(np.arange(NB), train_basins)] = np.nan


def windows(Ymask, basins):
    """Window end indices (basin, end) so every observed day after warm-up is covered once."""
    out = []
    for b in basins:
        t_obs = np.where(~np.isnan(Ymask[b]))[0]
        t_obs = t_obs[t_obs >= SEQ - PRED]
        if len(t_obs) == 0:
            continue
        ends = np.unique(np.minimum(t_obs - (t_obs - (SEQ - PRED)) % PRED + PRED - 1, NT - 1))
        out += [(b, e) for e in ends]
    return np.array(out)


def batch(win, Ysrc):
    b, e = win[:, 0], win[:, 1]
    t = e[:, None] - SEQ + 1 + np.arange(SEQ)[None, :]
    t = np.clip(t, 0, NT - 1)
    x = np.concatenate([C[b[:, None], t], D[t], Narr[b[:, None], t],
                        np.repeat(S[b][:, None, :], SEQ, 1)], -1)
    y = Ysrc[b[:, None], t].copy()
    y[:, : SEQ - PRED] = np.nan
    return torch.from_numpy(x), torch.from_numpy(y)


class Net(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.inp = nn.Linear(nin, HID)
        self.lstm = nn.LSTM(HID, HID, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(HID, 1)

    def forward(self, x):
        h, _ = self.lstm(torch.relu(self.inp(x)))
        return self.out(self.drop(h)).squeeze(-1)


tr_win = windows(Ytr, train_basins)
nin = NC + 2 + len(NVARS) + S.shape[1]
net = Net(nin)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
print(SPLIT, "train windows", len(tr_win), "inputs", nin, flush=True)
for ep in range(EPOCHS):
    net.train()
    perm = np.random.permutation(len(tr_win))[:PER_EPOCH]
    tl, t0 = [], time.time()
    for i in range(0, len(perm), BATCH):
        x, y = batch(tr_win[perm[i:i + BATCH]], Ytr)
        m = ~torch.isnan(y)
        p = net(x)
        loss = ((p[m] - y[m]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tl.append(loss.item())
    sched.step()
    print(f"epoch {ep} loss {np.mean(tl):.4f} {time.time() - t0:.0f}s", flush=True)

# predict test
Yte = np.where(test_tmask[None, :], Yn, np.nan)
Yte[np.setdiff1d(np.arange(NB), test_basins)] = np.nan
te_win = windows(Yte, test_basins)
net.eval()
rows = []
with torch.no_grad():
    for i in range(0, len(te_win), 512):
        w = te_win[i:i + 512]
        x, y = batch(w, Yte)
        p = net(x).numpy()
        yy = y.numpy()
        bb, tt = np.nonzero(~np.isnan(yy))
        tabs = w[bb, 1] - SEQ + 1 + tt
        rows.append(pd.DataFrame({"b": w[bb, 0], "t": tabs, "y": yy[bb, tt] * ysd + ymu, "pred": p[bb, tt] * ysd + ymu}))
pr = pd.concat(rows).drop_duplicates(["b", "t"])
pr["gid"] = [gids[b] for b in pr["b"]]
pr["year"] = dates.year.to_numpy()[pr["t"]]
r = evaluate(pr)
r.update(split=SPLIT, model="LSTM")
print(r, flush=True)
pr.to_parquet(RES / f"lstm_pred_{SPLIT}.parquet", index=False)
pd.DataFrame([r]).to_csv(RES / f"lstm_metrics_{SPLIT}.csv", index=False)
