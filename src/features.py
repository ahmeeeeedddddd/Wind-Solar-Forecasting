"""
features.py  –  Person A
Add lag features, rolling means, and calendar features to the cleaned DataFrame.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, month, and is_weekend columns."""
    df = df.copy()
    df["hour"]       = df.index.hour
    df["day_of_week"] = df.index.dayofweek          # 0=Mon, 6=Sun
    df["month"]      = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 24, 168]) -> pd.DataFrame:
    """
    Add lag columns for Production.
    Default lags (in hours):
      - 1h   : previous hour
      - 24h  : same hour yesterday
      - 168h : same hour last week
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df["Production"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [24, 168]) -> pd.DataFrame:
    """
    Add rolling mean and std of Production.
    Default windows (in hours): 24h and 168h (1 week).
    """
    df = df.copy()
    for w in windows:
        df[f"roll_mean_{w}h"] = df["Production"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}h"]  = df["Production"].shift(1).rolling(w).std()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in one call."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna()          # drop rows with NaN from lags/rolling
    return df


if __name__ == "__main__":
    from data_loading import load_raw, clean, merge_weather, chronological_split, save_processed

    df = load_raw()
    df = clean(df)
    df = merge_weather(df)
    df_features = build_features(df)

    print("Feature matrix shape:", df_features.shape)
    print("Columns:", df_features.columns.tolist())
    print(df_features.head())

    train, test = chronological_split(df_features)
    print(f"Train features: {len(train)} rows | Test features: {len(test)} rows")

    save_processed(df_features, "features_all.csv")
    save_processed(train,       "train_features.csv")
    save_processed(test,        "test_features.csv")

