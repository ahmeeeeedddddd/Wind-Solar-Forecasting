"""
data_loading.py  -  Person A
Load, clean, merge weather data, and do a chronological train/test split.
"""

import pandas as pd
import os

RAW_PATH_LOCAL   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Energy Production Dataset.csv")
RAW_PATH_DESKTOP = r"C:\Users\mhesh\Desktop\Energy Production Dataset.csv"
RAW_PATH         = RAW_PATH_LOCAL if os.path.exists(RAW_PATH_LOCAL) else RAW_PATH_DESKTOP
WEATHER_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "external", "france_weather.csv")
PROCESSED_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw() -> pd.DataFrame:
    """Read the raw CSV and return a tidy DataFrame with a proper datetime index."""
    df = pd.read_csv(RAW_PATH)

    # Build a proper datetime column from Date + Start_Hour
    df["datetime"] = pd.to_datetime(df["Date"], dayfirst=False) + pd.to_timedelta(df["Start_Hour"], unit="h")
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    # Drop columns that are redundant once we have the datetime index
    df = df.drop(columns=["Date", "End_Hour", "Day_of_Year", "Day_Name", "Month_Name", "Season"])

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and filter invalid rows."""
    # Drop duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep="first")]

    # Production must be non-negative
    df = df[df["Production"] >= 0]

    return df


def merge_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical France weather data (NASA POWER) into the production DataFrame.
    Weather columns added:
      - wind_speed_10m : wind speed at 10 m (m/s)
      - temp_2m        : air temperature at 2 m (degC)
      - irradiation    : surface solar irradiation (kW-hr/m^2/day)
      - cloud_cover    : cloud amount (%)

    If the weather file does not exist yet, returns df unchanged with a warning.
    """
    weather_path = os.path.abspath(WEATHER_PATH)
    if not os.path.exists(weather_path):
        print("[WARNING] Weather file not found. Skipping weather merge.")
        print(f"          Expected at: {weather_path}")
        return df

    weather = pd.read_csv(weather_path, index_col="datetime", parse_dates=True)

    # NASA POWER returns minute-level timestamps (YYYYMMDDHHNN where NN = minutes).
    # We need to aggregate to hourly means so timestamps align with production data.
    weather.index = weather.index.tz_localize(None)
    weather = weather.resample("h").mean()          # average any sub-hourly readings

    # Left-join on datetime index so production rows are never dropped
    merged = df.join(weather, how="left")

    # Forward-fill small gaps (e.g. missing hours at boundaries)
    weather_cols = ["wind_speed_10m", "temp_2m", "irradiation", "cloud_cover"]
    merged[weather_cols] = merged[weather_cols].ffill(limit=3)

    n_nan = merged[weather_cols].isnull().sum().sum()
    print(f"Weather merged. Total NaN in weather cols: {n_nan}")
    return merged


def chronological_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Simple time-based train / test split (no shuffling)."""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    return train, test


def save_processed(df: pd.DataFrame, filename: str):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    df = merge_weather(df)
    print("Shape after cleaning + weather merge:", df.shape)
    print(df.head())

    train, test = chronological_split(df)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    save_processed(df,    "cleaned.csv")
    save_processed(train, "train_raw.csv")
    save_processed(test,  "test_raw.csv")
