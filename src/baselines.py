"""
baselines.py  -  Person C
Naive and seasonal-naive baselines at horizons 1h, 6h, 24h.

Forecast for target time T, made at origin time T-h (h = horizon):
  - Naive                : y[T-h]   ("same as the last value we know")
  - Seasonal naive 24h   : y[T-24]  ("same hour yesterday")
  - Seasonal naive 168h  : y[T-168] ("same hour last week")

Run from the repo root:   python src/baselines.py
"""

import os
import pandas as pd

from evaluation import evaluate, evaluate_by_group, mase_scale, season_series

BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR    = os.path.join(BASE_DIR, "results", "tables")

TARGET   = "Production"
HORIZONS = [1, 6, 24]

BASELINES = {   # key -> display name
    "naive":      "Naive (last known value)",
    "snaive_24":  "Seasonal naive (same hour yesterday)",
    "snaive_168": "Seasonal naive (same hour last week)",
}


def load_splits():
    for f in ("train_features_clustered.csv", "test_features_clustered.csv"):
        if not os.path.exists(os.path.join(PROCESSED_DIR, f)):
            raise FileNotFoundError(
                f"data/processed/{f} not found. Run `git pull` first; if it is still missing, "
                "ask Person A/B to push data/processed/ (or send you the file).")
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features_clustered.csv"),
                        index_col="datetime", parse_dates=True)
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features_clustered.csv"),
                       index_col="datetime", parse_dates=True)
    return train, test


def regular_hourly_series(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Train+test Production on a strict hourly grid (missing hours become NaN),
    so shift(24) means exactly 24 hours."""
    full = pd.concat([train[TARGET], test[TARGET]]).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full.asfreq("h")


def baseline_prediction(full: pd.Series, key: str, horizon: int) -> pd.Series:
    if key == "naive":
        return full.shift(horizon)
    if key == "snaive_24":
        assert horizon <= 24, "daily seasonal naive needs horizon <= 24h"
        return full.shift(24)
    if key == "snaive_168":
        assert horizon <= 168, "weekly seasonal naive needs horizon <= 168h"
        return full.shift(168)
    raise ValueError(key)


def run_baselines():
    os.makedirs(TABLES_DIR, exist_ok=True)

    train, test = load_splits()
    full = regular_hourly_series(train, test)

    # ---- data sanity checks -------------------------------------------------
    print("Train:", train.index.min(), "->", train.index.max(), f"({len(train)} rows)")
    print("Test :", test.index.min(), "->", test.index.max(), f"({len(test)} rows)")
    print("Missing hourly timestamps in the full grid:", int(full.isna().sum()))

    scale = mase_scale(full.loc[:train.index.max()], m=24)
    print(f"MASE scale (train MAE of 24h seasonal naive): {scale:.2f} MW\n")

    y_test = full.reindex(test.index)
    seasons = season_series(test.index)
    clusters = test["cluster"] if "cluster" in test.columns else None

    rows, season_rows, cluster_rows = [], [], []
    for h in HORIZONS:
        for key, name in BASELINES.items():
            pred = baseline_prediction(full, key, h).reindex(test.index)

            r = evaluate(y_test, pred, scale)
            r.update(Model=name, Horizon_h=h)
            rows.append(r)

            s = evaluate_by_group(y_test, pred, seasons, scale, "Season")
            s["Model"], s["Horizon_h"] = name, h
            season_rows.append(s)

            if clusters is not None:
                c = evaluate_by_group(y_test, pred, clusters, scale, "Cluster")
                c["Model"], c["Horizon_h"] = name, h
                cluster_rows.append(c)

    cols = ["Model", "Horizon_h", "N", "MAE", "RMSE", "MASE"]
    overall = pd.DataFrame(rows)[cols].round({"MAE": 2, "RMSE": 2, "MASE": 4})
    overall.to_csv(os.path.join(TABLES_DIR, "baselines_results.csv"), index=False)

    by_season = pd.concat(season_rows)[["Model", "Horizon_h", "Season", "N", "MAE", "RMSE", "MASE"]]
    by_season.round({"MAE": 2, "RMSE": 2, "MASE": 4}).to_csv(
        os.path.join(TABLES_DIR, "baselines_by_season.csv"), index=False)

    if cluster_rows:
        by_cluster = pd.concat(cluster_rows)[["Model", "Horizon_h", "Cluster", "N", "MAE", "RMSE", "MASE"]]
        by_cluster.round({"MAE": 2, "RMSE": 2, "MASE": 4}).to_csv(
            os.path.join(TABLES_DIR, "baselines_by_cluster.csv"), index=False)

    print(overall.to_string(index=False))
    print("\nSaved: baselines_results.csv, baselines_by_season.csv, baselines_by_cluster.csv (results/tables)")
    return overall


if __name__ == "__main__":
    run_baselines()
