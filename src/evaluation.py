"""
evaluation.py  -  Person C
Metrics (MAE, RMSE, MASE) and helpers to slice errors by season and by cluster.
Shared by baselines.py, models.py and the final comparison notebook.
"""

import numpy as np
import pandas as pd

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def season_series(index: pd.DatetimeIndex) -> pd.Series:
    """Meteorological season for each timestamp (ordered categorical)."""
    s = pd.Series(index.month, index=index).map(SEASON_MAP)
    return pd.Series(pd.Categorical(s, categories=SEASON_ORDER, ordered=True), index=index)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mase_scale(y_train, m: int = 24) -> float:
    """
    MASE denominator = MAE of the seasonal-naive forecast (lag m) on the TRAIN set.
    Same definition as Person B (m=24), so all MASE numbers in the project are comparable.
    Pass a regular hourly series (gaps as NaN) so lag m really means m hours.
    """
    y = np.asarray(y_train, dtype=float)
    d = np.abs(y[m:] - y[:-m])
    d = d[~np.isnan(d)]
    return float(d.mean())


def evaluate(y_true, y_pred, scale: float) -> dict:
    """MAE / RMSE / MASE on the rows where both y_true and y_pred exist."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    m = mae(yt, yp)
    return {"N": int(mask.sum()), "MAE": m, "RMSE": rmse(yt, yp), "MASE": m / scale}


def evaluate_by_group(y_true: pd.Series, y_pred: pd.Series, groups: pd.Series,
                      scale: float, group_name: str = "group") -> pd.DataFrame:
    """Same metrics, computed separately for each value of `groups` (season, cluster...)."""
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
    rows = []
    for g, sub in df.groupby("g", sort=True, observed=True):
        r = evaluate(sub["y"], sub["p"], scale)
        r[group_name] = g
        rows.append(r)
    return pd.DataFrame(rows)[[group_name, "N", "MAE", "RMSE", "MASE"]]