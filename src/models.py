"""
models.py  -  Person C
Supervised forecasting at horizons 1h / 6h / 24h:
    baselines  vs  global ML models  vs  cluster-aware models
with hyper-parameter tuning via TimeSeriesSplit and error breakdown by season and cluster.

Run from the repo root:
    python src/models.py --quick     # fast check that everything works (a few minutes)
    python src/models.py             # full run with tuning (roughly 15-30 minutes)

Direct forecasting: a separate model is trained for each horizon h. For a target hour T the
model only sees information available at the forecast origin T-h:
  - last_known / lag_24h / lag_168h / rolling stats are shifted so they never look past T-h
  - calendar features of T and the weather at T (assumed to be a perfect weather forecast)
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import BASELINES, HORIZONS, TARGET, baseline_prediction, load_splits
from evaluation import evaluate, evaluate_by_group, mase_scale, season_series

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
PRED_DIR   = os.path.join(BASE_DIR, "results", "predictions")
WEATHER    = ["wind_speed_10m", "temp_2m", "irradiation", "cloud_cover"]
SEED       = 42


# --------------------------------------------------------------------------- features
def build_full_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train+test on a strict hourly grid (missing hours become NaN rows)."""
    full = pd.concat([train, test]).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full.asfreq("h")


def build_horizon_frame(full: pd.DataFrame, h: int) -> pd.DataFrame:
    """Feature matrix for horizon h. Every Production-based feature only uses data up to T-h."""
    y, idx = full[TARGET], full.index
    X = pd.DataFrame(index=idx)

    # known in advance
    X["hour"] = idx.hour
    X["day_of_week"] = idx.dayofweek
    X["month"] = idx.month
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    for c in WEATHER:                       # assumption: perfect weather forecast for hour T
        X[c] = full[c]

    # past production, available at the forecast origin T-h
    X["last_known"] = y.shift(h)
    if h < 24:                              # at h=24, lag_24h is identical to last_known
        X["lag_24h"] = y.shift(24)
    X["lag_168h"] = y.shift(168)
    for w in (24, 168):
        roll = y.shift(h).rolling(w, min_periods=(w * 3) // 4)
        X[f"roll_mean_{w}h"] = roll.mean()
        X[f"roll_std_{w}h"] = roll.std()

    # cluster information
    X["cluster"] = full["cluster"]          # same-day label (Person B). Uses the whole day -> optimistic
    dates = idx.normalize()
    daily = full["cluster"].groupby(dates).first()
    # label of day D-2: a day that is always fully finished at the forecast origin (leak-free)
    X["cluster_known"] = pd.Series(dates - pd.Timedelta(days=2), index=idx).map(daily)

    X["target"] = y
    X = X.dropna()
    X[["cluster", "cluster_known"]] = X[["cluster", "cluster_known"]].astype(int)
    return X


def feature_columns(frame: pd.DataFrame):
    return [c for c in frame.columns if c not in ("target", "cluster", "cluster_known")]


# --------------------------------------------------------------------------- tuning
def tune(estimator, space, kind, X, y, n_iter, n_splits):
    cv = TimeSeriesSplit(n_splits=n_splits)       # always trains on the past, validates on the future
    kw = dict(cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    if kind == "grid":
        search = GridSearchCV(estimator, space, **kw)
    else:
        search = RandomizedSearchCV(estimator, space, n_iter=n_iter, random_state=SEED, **kw)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_


def make_lgbm(**params):
    return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1, **params)


# --------------------------------------------------------------------------- main run
def run(quick: bool = False):
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    t0 = time.time()

    train, test = load_splits()
    full = build_full_frame(train, test)
    train_end, test_start = train.index.max(), test.index.min()
    scale = mase_scale(full.loc[:train_end, TARGET], m=24)
    print(f"Train until {train_end} | Test from {test_start} | MASE scale = {scale:.2f} MW")

    n_splits = 2 if quick else 3
    rf_space = {"n_estimators": [50] if quick else [100, 200], "max_depth": [8, 12, 16, None],
                "min_samples_leaf": [1, 5, 20], "max_features": [0.5, 0.8, 1.0]}
    lgbm_space = {"n_estimators": [100] if quick else [200, 400, 800],
                  "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63],
                  "min_child_samples": [20, 50, 100], "subsample": [0.8, 1.0],
                  "colsample_bytree": [0.7, 1.0]}
    rf_iter, lgbm_iter = (2, 3) if quick else (6, 12)

    overall, by_season, by_cluster, best_params = [], [], [], []

    for h in HORIZONS:
        print(f"\n===== Horizon {h}h  ({(time.time() - t0) / 60:.1f} min elapsed) =====")
        frame = build_horizon_frame(full, h)
        tr, te = frame.loc[:train_end], frame.loc[test_start:]
        cols = feature_columns(frame)
        Xtr, ytr, Xte, yte = tr[cols], tr["target"], te[cols], te["target"]
        print(f"features ({len(cols)}): {cols}\ntrain rows {len(tr)} | test rows {len(te)}")

        preds = {}   # name -> (group, predictions aligned to te.index)

        # 1) baselines (same rows as the models, so the comparison is fair)
        for key, name in BASELINES.items():
            preds[name] = ("Baseline", baseline_prediction(full[TARGET], key, h).reindex(te.index))

        # 2) global models (no cluster information)
        lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
        preds["Linear Regression"] = ("Global", pd.Series(lin.predict(Xte), index=te.index))

        ridge, p, cv_mae = tune(make_pipeline(StandardScaler(), Ridge()),
                                {"ridge__alpha": [0.01, 0.1, 1, 10, 100, 1000]}, "grid", Xtr, ytr, 0, n_splits)
        preds["Ridge"] = ("Global", pd.Series(ridge.predict(Xte), index=te.index))
        best_params.append({"Horizon_h": h, "Model": "Ridge", "CV_MAE": round(cv_mae, 2), "Params": str(p)})
        print(f"Ridge done      | CV MAE {cv_mae:.1f}")

        rf, p, cv_mae = tune(RandomForestRegressor(random_state=SEED, n_jobs=-1), rf_space,
                             "random", Xtr, ytr, rf_iter, n_splits)
        preds["Random Forest"] = ("Global", pd.Series(rf.predict(Xte), index=te.index))
        best_params.append({"Horizon_h": h, "Model": "Random Forest", "CV_MAE": round(cv_mae, 2), "Params": str(p)})
        print(f"Random Forest done | CV MAE {cv_mae:.1f} | {p}")

        lgbm, lgbm_p, cv_mae = tune(make_lgbm(), lgbm_space, "random", Xtr, ytr, lgbm_iter, n_splits)
        preds["LightGBM"] = ("Global", pd.Series(lgbm.predict(Xte), index=te.index))
        best_params.append({"Horizon_h": h, "Model": "LightGBM", "CV_MAE": round(cv_mae, 2), "Params": str(lgbm_p)})
        print(f"LightGBM done   | CV MAE {cv_mae:.1f} | {lgbm_p}")

        # ablation: does the (ideal) weather information matter?
        no_w = [c for c in cols if c not in WEATHER]
        m = make_lgbm(**lgbm_p).fit(tr[no_w], ytr)
        preds["LightGBM (no weather)"] = ("Global", pd.Series(m.predict(te[no_w]), index=te.index))

        # 3) cluster-aware models (LightGBM with the tuned parameters)
        pred_a = pd.Series(np.nan, index=te.index)             # (a) one model per cluster
        for c in sorted(tr["cluster"].unique()):
            m_tr, m_te = tr["cluster"] == c, te["cluster"] == c
            if m_te.sum() == 0:
                continue
            mc = make_lgbm(**lgbm_p).fit(tr.loc[m_tr, cols], tr.loc[m_tr, "target"])
            pred_a[m_te] = mc.predict(te.loc[m_te, cols])
        preds["LightGBM + per-cluster models (a)"] = ("Cluster-aware", pred_a)

        mb = make_lgbm(**lgbm_p).fit(tr[cols + ["cluster"]], ytr)               # (b) cluster as feature
        preds["LightGBM + cluster feature (b)"] = ("Cluster-aware",
                                                   pd.Series(mb.predict(te[cols + ["cluster"]]), index=te.index))
        mk = make_lgbm(**lgbm_p).fit(tr[cols + ["cluster_known"]], ytr)         # (b) leak-free label
        preds["LightGBM + cluster feature (b, leak-free)"] = (
            "Cluster-aware", pd.Series(mk.predict(te[cols + ["cluster_known"]]), index=te.index))

        # evaluation: overall / by season / by cluster
        seasons = season_series(te.index)
        for name, (group, pred) in preds.items():
            r = evaluate(yte, pred, scale)
            r.update(Group=group, Model=name, Horizon_h=h)
            overall.append(r)
            s = evaluate_by_group(yte, pred, seasons, scale, "Season")
            s["Group"], s["Model"], s["Horizon_h"] = group, name, h
            by_season.append(s)
            c = evaluate_by_group(yte, pred, te["cluster"], scale, "Cluster")
            c["Group"], c["Model"], c["Horizon_h"] = group, name, h
            by_cluster.append(c)

        out = pd.DataFrame({"y_true": yte, "cluster": te["cluster"]})
        for name, (_, pred) in preds.items():
            out[name] = pred
        out.to_csv(os.path.join(PRED_DIR, f"predictions_h{h}.csv"))

    # ---- save tables
    rd = {"MAE": 2, "RMSE": 2, "MASE": 4}
    head = ["Group", "Model", "Horizon_h"]
    final = pd.DataFrame(overall)[head + ["N", "MAE", "RMSE", "MASE"]].round(rd)
    final.to_csv(os.path.join(TABLES_DIR, "final_comparison.csv"), index=False)
    pd.concat(by_season)[head + ["Season", "N", "MAE", "RMSE", "MASE"]].round(rd).to_csv(
        os.path.join(TABLES_DIR, "final_by_season.csv"), index=False)
    pd.concat(by_cluster)[head + ["Cluster", "N", "MAE", "RMSE", "MASE"]].round(rd).to_csv(
        os.path.join(TABLES_DIR, "final_by_cluster.csv"), index=False)
    pd.DataFrame(best_params).to_csv(os.path.join(TABLES_DIR, "best_hyperparameters.csv"), index=False)

    # ---- print summary
    for h in HORIZONS:
        print(f"\n=== Horizon {h}h (sorted by MAE) ===")
        print(final[final.Horizon_h == h].sort_values("MAE")
              [["Group", "Model", "MAE", "RMSE", "MASE"]].to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Tables -> results/tables, predictions -> results/predictions")
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, just to check the pipeline")
    run(quick=ap.parse_args().quick)
