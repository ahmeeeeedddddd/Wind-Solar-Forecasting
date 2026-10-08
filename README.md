# Wind & Solar Energy Forecasting (France)

An end-to-end machine learning project to forecast hourly wind and solar energy production in France using historical generation data and external meteorological features (NASA POWER).

---

## 📁 Project Structure (Current State: Person A & Person B Complete)

```
wind-solar-forecasting/
├── README.md                                  # Project overview and instructions
├── requirements.txt                           # Project dependencies
├── data/
│   ├── raw/                                   # Original dataset (local immutable, not pushed)
│   │   └── Energy Production Dataset.csv
│   ├── external/                              # Historical France weather data (NASA POWER)
│   │   └── france_weather.csv                 # 51,864 hourly records (2020-2025)
│   └── processed/                             # Processed datasets for modeling
│       ├── cleaned.csv                        # Deduplicated & sorted data
│       ├── train_raw.csv                      # Chronological train split (80%)
│       ├── test_raw.csv                       # Chronological test split (20%)
│       ├── features_all.csv                   # Full feature matrix (18 columns, 0 nulls)
│       ├── train_features.csv                 # Training feature matrix (41,352 rows)
│       ├── test_features.csv                  # Test feature matrix (10,338 rows)
│       ├── cluster_labels.csv                 # Daily profile cluster assignments (KMeans, GMM, Hierarchical)
│       ├── train_features_clustered.csv       # Training feature set with cluster label column
│       └── test_features_clustered.csv        # Test feature set with cluster label column
├── notebooks/
│   ├── 01_eda.ipynb                           # Person A: Exploratory Data Analysis
│   ├── 02_weather_and_features.ipynb          # Person A: Weather merge & feature validation
│   └── 03_clustering.ipynb                    # Person B: Daily profiles, clustering & cluster-aware experiments
├── src/
│   ├── data_loading.py                        # Person A: Load, clean, merge weather, split
│   ├── features.py                            # Person A: Lag, rolling stats, calendar features
│   ├── clustering.py                          # Person B: Daily profile matrix, K-Means/GMM/Hierarchical, PCA, experiments
│   └── plotting.py                            # Shared figure styling and color palette
├── results/
│   ├── figures/                               # Exported EDA, clustering & feature plots (PNG)
│   └── tables/                                # Summary & experimental results tables (CSV)
│       ├── features_summary.csv
│       ├── clustering_k_selection.csv
│       ├── clustering_algorithms_comparison.csv
│       ├── cluster_summary.csv
│       ├── cluster_aware_experiments_summary.csv
│       └── cluster_experiment_a_per_cluster.csv
├── report/
│   ├── person_a_report.md                     # Comprehensive Person A technical report
│   └── person_b_report.md                     # Comprehensive Person B technical report
└── presentation/
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Person A)
To reproduce the cleaning, weather merging, feature engineering, and train/test splits:
```bash
python src/data_loading.py
python src/features.py
```

### 3. Run Unsupervised Learning & Clustering Pipeline (Person B)
To construct daily profile matrices, perform diagnostic cluster selection, fit K-Means/GMM/Hierarchical models, generate visualizations, export cluster labels, and run cluster-aware forecasting experiments:
```bash
python src/clustering.py
```

### 4. Explore Notebooks
Open Jupyter and run:
- `notebooks/01_eda.ipynb` for full exploratory analysis and data visualizations.
- `notebooks/02_weather_and_features.ipynb` for feature correlation and lag analysis.
- `notebooks/03_clustering.ipynb` for unsupervised profile clustering and cluster-aware forecasting experiments.

---

## 👤 Team Member Roles & Status

| Role | Responsibility | Status | Primary Output Files |
|---|---|---|---|
| **Person A** | Data loading, cleaning, France weather integration, feature engineering, EDA, initial report | **Completed (100%)** | `src/data_loading.py`, `src/features.py`, `src/plotting.py`, `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`, `data/processed/*`, `report/person_a_report.md` |
| **Person B** | Daily profiles matrix, k-means/GMM/Hierarchical clustering, silhouette & elbow analysis, cluster labels, PCA, cluster-aware experiments | **Completed (100%)** | `src/clustering.py`, `notebooks/03_clustering.ipynb`, `data/processed/cluster_labels.csv`, `data/processed/*clustered.csv`, `results/figures/cluster_*.png`, `report/person_b_report.md` |
| **Person C** | Baselines (naive, seasonal naive), ML models (Ridge, RF, LightGBM/XGBoost), TimeSeriesSplit CV, horizon evaluation | *Pending* | Inputs ready in `data/processed/train_features_clustered.csv` & `test_features_clustered.csv` |
