# Person A Report — Data, Weather, and Features

## 1. Introduction

This project develops an end-to-end forecasting pipeline for hourly **wind and solar energy production** in France. The dataset encompasses **51,864 hourly observations** recorded between **January 1, 2020 and November 30, 2025** (~5.9 years).

As designated for **Person A**, this phase establishes the data foundation:
1. Loading and auditing the raw Kaggle dataset.
2. Determining target representation and data schema.
3. Timestamp harmonization, deduplication, and quality validation.
4. Comprehensive Exploratory Data Analysis (EDA) of diurnal, weekly, and seasonal dynamics.
5. Acquiring external historical France weather data (wind speed, solar irradiation, air temperature, cloud cover) from NASA POWER and merging by timestamp.
6. Engineering lag structures, rolling statistics, and calendar features.
7. Constructing chronological train/test splits and exporting standardized datasets for **Person B** (clustering) and **Person C** (supervised forecasting).

---

## 2. Data Description & Schema Audit

### 2.1 Dataset Overview

The dataset provides hourly energy generation metrics in tabular format:

| Column | Raw Type | Description | Handled As |
|---|---|---|---|
| `Date` | string (`MM/DD/YYYY`) | Calendar date of observation | Combined into `datetime` index |
| `Start_Hour` | integer (`0–23`) | Hour of observation start | Combined into `datetime` index |
| `End_Hour` | integer (`0–23`) | Hour of observation conclusion | Redundant; dropped |
| `Source` | string | Generation category (`Wind`, `Solar`, `Mixed`) | Feature / filter variable |
| `Day_of_Year` | integer (`1–366`) | Ordinal day | Derived from index |
| `Day_Name` | string | Weekday name | Derived as numeric `day_of_week` |
| `Month_Name` | string | Month name | Derived as numeric `month` |
| `Season` | string | Season label | Derived from month |
| `Production` | integer | Generated energy (MWh / arbitrary units) | **Target variable** |

### 2.2 Target Formulation Decision

*Evaluation of Target Definition*:
- Rather than providing separate concurrent columns for Wind and Solar generation simultaneously at each hour, the dataset presents a single `Production` target with an associated `Source` indicator.
- Analysis reveals that `Solar` is recorded almost exclusively during daylight hours (hours 7 to 20), whereas `Wind` is recorded continuously 24/7 across the entire year.
- **Decision**: We preserve `Production` as the unified continuous target while retaining `Source` and its one-hot/label encoded representation to allow both unified multi-source models and source-segregated models.

### 2.3 Source Breakdown & Generation Shares

Audit of the cleaned dataset (51,858 records):

| Energy Source | Observation Count | Percentage of Rows | Cumulative Generation | Share of Total Production |
|---|---|---|---|---|
| **Wind** | 42,478 | 81.91% | 267,954,399 units | **83.14%** |
| **Solar** | 9,378 | 18.08% | 54,334,685 units | **16.86%** |
| **Mixed** | 2 | <0.01% | 3,474 units | **<0.01%** |

### 2.4 Integrity & Quality Checks

- **Missing Values**: 0 null entries across all columns.
- **Duplicate Timestamps**: 6 duplicate timestamp entries identified and dropped (keeping first occurrence).
- **Physical Feasibility**: 0 negative production values; production minimum is 0.

---

## 3. Preprocessing & Temporal Splitting

### 3.1 Datetime Construction

Timestamps were consolidated to strict hourly `datetime64[ns]` index:
```python
df["datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(df["Start_Hour"], unit="h")
df = df.sort_values("datetime").set_index("datetime")
```

### 3.2 Chronological Train/Test Partitioning

To avoid data leakage inherent in random shuffling of time series:
- **Partition Ratio**: 80% Train, 20% Test.
- **Train Window**: 2020-01-01 00:00:00 to ~2024-09-26 04:00:00 (41,352 observations in feature matrix).
- **Test Window**: 2024-09-26 05:00:00 to 2025-11-30 23:00:00 (10,338 observations in feature matrix).

---

## 4. External Historical Weather Data Integration

Historical meteorological data for France was obtained from the **NASA Prediction of Worldwide Energy Resources (POWER) API** at geographic centroid coordinates (Latitude: 46.2276°N, Longitude: 2.2137°E) covering the identical 2020–2025 temporal range:

| Variable | API Parameter | Physical Units | Relevance to Energy Production |
|---|---|---|---|
| **Wind Speed (10m)** | `WS10M` | m/s | Primary physical driver for wind turbine kinetic power ($P \propto v^3$) |
| **Solar Irradiation** | `ALLSKY_SFC_SW_DWN` | kW-hr/m²/day | Direct insolation driving photovoltaic cell excitation |
| **Temperature (2m)** | `T2M` | °C | Modulates PV panel efficiency (negative thermal coefficient) |
| **Cloud Cover** | `CLOUD_AMT` | % | Attenuates solar irradiance reaching panels |

The meteorological series was aligned onto the production time index via hourly timestamp merging and exported to `data/external/france_weather.csv` (51,864 hours, 0 missing values).

---

## 5. Feature Engineering

Features are generated systematically in `src/features.py`:

### 5.1 Calendar & Temporal Features
- `hour`: Hour of day (0–23) — captures diurnal solar cycle and daily wind pattern.
- `day_of_week`: Day of week (0=Monday, 6=Sunday).
- `month`: Month of year (1–12) — captures macro seasonal variations.
- `is_weekend`: Binary flag (1 if Saturday or Sunday, else 0).

### 5.2 Autoregressive Lag Features
- `lag_1h`: Immediate persistence baseline ($t-1$).
- `lag_24h`: Daily seasonal persistence ($t-24$).
- `lag_168h`: Weekly seasonal persistence ($t-168$).

### 5.3 Rolling Window Statistics
- `roll_mean_24h` & `roll_std_24h`: 24-hour moving mean and volatility (shifted by 1 hour to prevent lookahead bias).
- `roll_mean_168h` & `roll_std_168h`: 7-day moving mean and volatility (shifted by 1 hour).

### 5.4 Feature Matrix Summary
The resulting dataset (`data/processed/features_all.csv`) contains **51,690 observations** (after 168h warmup drop) and **18 columns** with zero null values.

Summary statistics are exported in `results/tables/features_summary.csv`.

---

## 6. Key EDA Findings & Visualizations

Figures saved in `results/figures/`:

1. **Diurnal Cycle** (`01_hourly_avg.png`):
   - Solar production displays a bell curve strictly concentrated between 07:00 and 20:00, peaking at ~13:00.
   - Wind generation shows a steady 24-hour profile with a slight nocturnal increase due to atmospheric boundary layer stabilization.

2. **Distribution & Skewness** (`01_distribution.png`):
   - Solar distribution is zero-inflated (nighttime) with a right tail up to ~14,000 units.
   - Wind generation exhibits a Weibull-like distribution spanning 0 to ~22,000 units.

3. **Seasonality** (`01_seasonal_box.png` & `01_monthly.png`):
   - Solar generation exhibits high summer peaks (June–August) and low winter output (December–January).
   - Wind generation exhibits peak production during winter storm systems (November–February) and milder generation during summer.

4. **Weather Correlation** (`02_correlation.png` & `02_weather_scatter.png`):
   - Strong positive correlation between `irradiation` and solar production during daylight.
   - Positive correlation between `wind_speed_10m` and wind production.
   - High correlation between `Production` and autoregressive lags `lag_1h` ($r \approx 0.94$), confirming strong temporal persistence suitable for autoregressive forecasting.

---

## 7. Deliverables & Interfaces for Downstream Tasks

Person A provides clean, validated inputs for downstream tasks:

### 7.1 Interface for Person B (Unsupervised Learning & Clustering)
- **Input File**: `data/processed/train_features.csv` (or `data/processed/train_raw.csv`).
- **Structure**: Continuous hourly series ready to pivot into a **daily profile matrix** ($N_{\text{days}} \times 24_{\text{hours}}$).
- **Cluster Target**: Daily shapes can be clustered with K-Means, GMM, or Hierarchical clustering, and evaluated via Silhouette and Elbow methods.

### 7.2 Interface for Person C (Supervised Models & Evaluation)
- **Input Files**: `data/processed/train_features.csv` (training) and `data/processed/test_features.csv` (evaluation).
- **Available Features**: Calendar variables (`hour`, `day_of_week`, `month`, `is_weekend`), meteorological drivers (`wind_speed_10m`, `irradiation`, `temp_2m`, `cloud_cover`), autoregressive lags (`lag_1h`, `lag_24h`, `lag_168h`), and rolling statistics (`roll_mean_24h`, `roll_std_24h`, `roll_mean_168h`, `roll_std_168h`).
- **Splitting Strategy**: Pre-split chronologically to enforce zero lookahead bias with standard `TimeSeriesSplit` cross-validation.

### 7.3 Deliverables Summary
- [x] `src/data_loading.py`: Robust data ingestion, cleaning, weather integration, and chronological train/test splitting.
- [x] `src/features.py`: Feature construction for lags, rolling window statistics, and calendar features.
- [x] `src/plotting.py`: Shared figure styling module with unified color palettes and publication-ready parameters.
- [x] `data/external/france_weather.csv`: 51,864 hourly observations of historical France weather metrics from NASA POWER.
- [x] `data/processed/`: Standardized CSV exports (`cleaned.csv`, `train_raw.csv`, `test_raw.csv`, `features_all.csv`, `train_features.csv`, `test_features.csv`).
- [x] `notebooks/01_eda.ipynb`: Exploratory Data Analysis notebook.
- [x] `notebooks/02_weather_and_features.ipynb`: Weather integration and feature engineering analysis notebook.
- [x] `results/figures/`: 8 exported high-resolution PNG plots.
- [x] `results/tables/features_summary.csv`: Summary statistics for all engineered features.
- [x] `requirements.txt`: Minimal dependencies for full reproducibility.

