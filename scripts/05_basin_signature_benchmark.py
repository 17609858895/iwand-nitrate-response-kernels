"""20-algorithm benchmark + stacking for ungauged-basin nitrate signatures (one row per basin).

Targets: log long-term median nitrate (default) or other summary signatures.
Validation: 5-fold spatial-block CV (60 k-means blocks on gauge coordinates); stacking meta-learner
is trained on inner out-of-fold predictions only.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression,
                                  Ridge, RidgeCV)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from common import EPS, NVARS, RES, load, static_features

warnings.filterwarnings("ignore")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "50th(mg-N/l)"

obs, st, by = load()
SF = static_features(st)
# basin-period mean N inputs and climate over each basin's own sampled years
yrs = obs.groupby("gid")["year"].agg(["min", "max"])
byp = by.merge(yrs, left_on="gid", right_index=True)
byp = byp[(byp["year"] >= byp["min"]) & (byp["year"] <= byp["max"])]
agg = byp.groupby("gid")[NVARS + ["P", "T", "PET", "MELT", "P_PET"]].mean().add_prefix("mean_")
d = st.set_index("gid")[SF + [TARGET]].join(agg)
y = np.log(d[TARGET].to_numpy() + EPS)
X = d.drop(columns=[TARGET])
FEATS = X.columns.tolist()
X = X.to_numpy(np.float64)
blk = KMeans(60, n_init=5, random_state=0).fit_predict(d[["Lat", "Lon"]])
outer = list(GroupKFold(5).split(X, y, groups=blk))

lin = lambda m: make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal"), StandardScaler(), m)
tree = lambda m: make_pipeline(SimpleImputer(strategy="median"), m)
MODELS = {
    "OLS": lin(LinearRegression()),
    "Ridge": lin(Ridge(alpha=10)),
    "Lasso": lin(Lasso(alpha=0.005, max_iter=20000)),
    "ElasticNet": lin(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
    "BayesRidge": lin(BayesianRidge()),
    "Huber": lin(HuberRegressor(alpha=1e-2, max_iter=2000)),
    "PLS": lin(PLSRegression(n_components=12)),
    "KNN": lin(KNeighborsRegressor(n_neighbors=10, weights="distance")),
    "SVR": lin(SVR(C=3.0, epsilon=0.05, gamma="scale")),
    "KernelRidge": lin(KernelRidge(alpha=0.3, kernel="rbf", gamma=0.005)),
    "GPR": lin(GaussianProcessRegressor(ConstantKernel() * RBF(10.0) + WhiteKernel(0.3), normalize_y=True)),
    "CART": tree(DecisionTreeRegressor(min_samples_leaf=10, random_state=0)),
    "RF": tree(RandomForestRegressor(500, min_samples_leaf=2, max_features=0.4, n_jobs=8, random_state=0)),
    "ExtraTrees": tree(ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.5, n_jobs=8, random_state=0)),
    "AdaBoost": tree(AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=300, learning_rate=0.05, random_state=0)),
    "GBDT": tree(GradientBoostingRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, random_state=0)),
    "HistGB": HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
    "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.6, n_jobs=8, random_state=0),
    "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, n_jobs=8, verbose=-1, random_state=0),
    "CatBoost": CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, verbose=0, random_seed=0, thread_count=8),
    "MLP": lin(MLPRegressor((256, 128), alpha=1e-2, learning_rate_init=1e-3, max_iter=600, early_stopping=True, random_state=0)),
}


def fit_pred(m, Xtr, ytr, Xte):
    m.fit(Xtr, ytr)
    return np.ravel(m.predict(Xte))


def metrics(o, p):
    return {"R2": 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2), "RMSE": float(np.sqrt(np.mean((o - p) ** 2))),
            "MAE": float(np.mean(np.abs(o - p)))}


oof = {k: np.zeros(len(y)) for k in MODELS}
oof["Stacking"] = np.zeros(len(y))
fold_rows, t0 = [], time.time()
for f, (tr, te) in enumerate(outer):
    inner = list(GroupKFold(4).split(X[tr], y[tr], groups=blk[tr]))
    inner_oof = np.zeros((len(tr), len(MODELS)))
    for j, (k, m) in enumerate(MODELS.items()):
        oof[k][te] = fit_pred(m, X[tr], y[tr], X[te])
        fold_rows.append({"fold": f, "model": k, **metrics(y[te], oof[k][te])})
        for itr, ite in inner:
            inner_oof[ite, j] = fit_pred(m, X[tr][itr], y[tr][itr], X[tr][ite])
    meta = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=True).fit(inner_oof, y[tr])
    te_base = np.column_stack([oof[k][te] for k in MODELS])
    oof["Stacking"][te] = meta.predict(te_base)
    fold_rows.append({"fold": f, "model": "Stacking", **metrics(y[te], oof["Stacking"][te])})
    print(f"fold {f} done {time.time() - t0:.0f}s; stacking R2 {fold_rows[-1]['R2']:.3f}", flush=True)

summ = pd.DataFrame([{"model": k, **metrics(y, v)} for k, v in oof.items()]).sort_values("R2", ascending=False)
folds = pd.DataFrame(fold_rows)
summ = summ.merge(folds.groupby("model")["R2"].agg(R2_fold_mean="mean", R2_fold_sd="std").reset_index(), on="model")
tag = TARGET.replace("(mg-N/l)", "").replace("th", "")
summ.to_csv(RES / f"basin_benchmark_{tag}.csv", index=False)
folds.to_csv(RES / f"basin_benchmark_folds_{tag}.csv", index=False)
pd.DataFrame({"gid": d.index, "y": y, **oof}).to_csv(RES / f"basin_benchmark_oof_{tag}.csv", index=False)
print(summ.round(3).to_string(index=False))
