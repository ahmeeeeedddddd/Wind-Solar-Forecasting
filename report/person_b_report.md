# Person B Technical Report: Unsupervised Learning & Cluster-Aware Forecasting

## Executive Summary
This report presents the implementation and findings for **Person B: Unsupervised Learning and Cluster-Aware Forecasting** on the France Wind & Solar Energy Production dataset. The core objective is to identify distinct daily production patterns, interpret their meteorological and seasonal dynamics, and leverage daily cluster labels to enhance hourly energy forecasting performance.

---

## 1. Clustering Methodology

### 1.1 Daily Profile Matrix Construction
Hourly production records (51,858 time points across 2020–2025) were reshaped into a daily profile matrix $X \in \mathbb{R}^{N \times 24}$, where each row represents a unique calendar date ($N = 2,155$ full days) and columns correspond to the 24 hourly production readings ($00:00$ to $23:00$). Incomplete days with missing hours were removed to ensure feature alignment.

### 1.2 Normalization
Each hourly feature column was standardized using `StandardScaler` (zero mean, unit variance):
$$z_{d, h} = \frac{x_{d, h} - \mu_h}{\sigma_h}$$
This normalization ensures that clustering reflects structural daily profile shapes and hourly variances without being dominated by extreme scale differences.

### 1.3 Cluster Count ($k$) Selection
We evaluated $k \in [2, 10]$ using two complementary diagnostic metrics:
- **Elbow Method (Inertia)**: Measured total within-cluster sum of squares.
- **Silhouette Score**: Quantified intra-cluster cohesion versus inter-cluster separation.

| $k$ | Inertia | Silhouette Score | Recommendation |
|---|---|---|---|
| **2** | 35,412.8 | 0.384 | High separation, but merges transitional profiles |
| **3** | **27,105.4** | **0.342** | **Optimal balance of inertia elbow & interpretability** |
| 4 | 22,891.2 | 0.301 | Subdivides summer peak into redundant groups |
| 5 | 19,784.6 | 0.279 | Overfitting noise |

**Decision**: $k = 3$ was selected as the optimal cluster count.

### 1.4 Clustering Algorithms Comparison
Three distinct clustering algorithms were fitted on the standardized profile matrix with $k = 3$:

1. **K-Means ($k=3$)**: Partitioning around centroids (n_init=20).
2. **Gaussian Mixture Model (GMM)**: Probabilistic soft-clustering with full covariance matrix.
3. **Hierarchical Agglomerative Clustering**: Bottom-up merging using **Ward's linkage** criterion.

| Algorithm | Silhouette Score | Key Characteristic |
|---|---|---|
| **K-Means** | **0.342** | Convex, crisp boundaries, fast convergence |
| **GMM** | 0.318 | Ellipsoidal, handles cluster overlap |
| **Hierarchical (Ward)** | 0.339 | Clear hierarchical tree, minimal variance increase |

K-Means cluster assignments were selected as the primary labels for downstream forecasting experiments due to superior cohesion and stability.

---

## 2. Cluster Analysis & Interpretation

### 2.1 Profile Interpretation & Named Clusters
Centroid analysis of the un-scaled hourly profiles revealed 3 operational regimes:

1. **Cluster 0: Low Production (Winter / Overcast)**  
   - **Characteristics**: Low generation throughout the day (mean daily total: 89,105 MW). Flat solar curve due to cloud cover and short daylight.
   - **Frequency**: 1,121 days (52.02% of total dataset).
   - **Peak Hour**: 13:00 (5,952 MW).

2. **Cluster 1: High Production (Summer / Sunny)**  
   - **Characteristics**: Sharp bell-shaped solar curve peaking during midday (mean daily total: 302,637 MW). High clear-sky irradiation.
   - **Frequency**: 301 days (13.97% of total dataset).
   - **Peak Hour**: 13:00 (15,494 MW).

3. **Cluster 2: Moderate Production (Transitional)**  
   - **Characteristics**: Intermediate generation profile (mean daily total: 177,555 MW), typical of spring/autumn months or mixed cloud cover.
   - **Frequency**: 733 days (34.01% of total dataset).
   - **Peak Hour**: 13:00 (10,103 MW).

### 2.2 Seasonal & Monthly Dynamics
Cross-tabulating cluster membership across calendar months confirmed strong seasonal alignment:
- **Summer (June–August)**: Dominated by **Cluster 1 (High Production)** (~60–75% of summer days).
- **Winter (November–February)**: Overwhelmingly **Cluster 0 (Low Production)** (>85% of winter days).
- **Spring/Autumn (April, May, September, October)**: Primarily **Cluster 2 (Transitional)**.

### 2.3 PCA Dimensionality Reduction
Principal Component Analysis (PCA) reduced the 24-dimensional profile space into 2 principal components:
- **PC1 (Variance Explained: 68.4%)**: Captures overall daily total energy volume.
- **PC2 (Variance Explained: 14.2%)**: Captures peak-to-base ratio and solar curve sharpness.
- **Total Variance Explained**: **82.6%**.
- 2D scatter plots show distinct spatial clusters along PC1 with clean separation between Cluster 0 (left) and Cluster 1 (right).

---

## 3. Cluster-Aware Forecasting Results

To evaluate whether unsupervised daily profile labels improve hourly forecasting accuracy, we conducted two experiments against a **Global Random Forest Baseline** using the exact chronological split (80% train / 20% test).

### 3.1 Experimental Setup & Evaluation Metrics
- **Baseline**: Single Random Forest model trained on standard features (`lags`, `rolling statistics`, `calendar features`).
- **Experiment (a) — Sub-Models**: $k=3$ separate Random Forest models trained independently on data from Cluster 0, 1, and 2. Test predictions routed based on test day cluster assignment.
- **Experiment (b) — Cluster Feature**: Single global Random Forest model trained with `cluster` label appended as an extra categorical/integer feature.

Metrics evaluated:
- **MAE** (Mean Absolute Error, MW)
- **RMSE** (Root Mean Squared Error, MW)
- **MASE** (Mean Absolute Scaled Error, relative to 24h naive baseline)

### 3.2 Performance Summary

| Model Strategy | MAE (MW) | RMSE (MW) | MASE | MAE Improvement vs. Baseline |
|---|---|---|---|---|
| **Global RF Baseline (No Cluster)** | 509.88 | 868.51 | 0.2040 | — |
| **Experiment (a): Separate Model per Cluster** | 514.25 | 862.63 | 0.2058 | +4.37 MW (+0.86% error) |
| **Experiment (b): Cluster Label as Extra Feature** | **505.47** | **864.61** | **0.2023** | **-4.41 MW (-0.86% error)** |

### 3.3 Experiment (a) Per-Cluster Breakdown

| Cluster | Test Sample Count | MAE (MW) | RMSE (MW) |
|---|---|---|---|
| **Cluster 0 (Low Production)** | 5,424 hours | 382.14 | 631.20 |
| **Cluster 1 (High Production)** | 1,488 hours | 812.45 | 1,289.10 |
| **Cluster 2 (Moderate Production)**| 3,426 hours | 592.30 | 974.85 |

### 3.4 Key Findings & Insights
1. **Cluster Feature Integration (Exp b)** yields the best overall performance, reducing MAE by **4.41 MW** and achieving the lowest MASE (**0.2023**).
2. **Sub-Model Partitioning (Exp a)** slightly degrades global MAE compared to the baseline (+4.37 MW), because splitting training data into sub-clusters reduces the sample size available to each individual model (especially Cluster 1 with only 301 days).
3. **Conclusion for Person C Integration**: The cluster label column created by Person B (`data/processed/train_features_clustered.csv`) should be provided to Person C as an engineered feature for XGBoost/LightGBM global models.

---

## Deliverable Artifacts Summary

- **Source Code**: `src/clustering.py`
- **Notebook**: `notebooks/03_clustering.ipynb`
- **Generated Datasets**:
  - `data/processed/cluster_labels.csv`
  - `data/processed/train_features_clustered.csv`
  - `data/processed/test_features_clustered.csv`
- **Figures**:
  - `results/figures/elbow_silhouette.png`
  - `results/figures/clustering_algorithms_comparison.png`
  - `results/figures/dendrogram.png`
  - `results/figures/cluster_profiles.png`
  - `results/figures/cluster_monthly_distribution.png`
  - `results/figures/pca_clusters.png`
- **Tables**:
  - `results/tables/clustering_k_selection.csv`
  - `results/tables/clustering_algorithms_comparison.csv`
  - `results/tables/cluster_summary.csv`
  - `results/tables/cluster_aware_experiments_summary.csv`
  - `results/tables/cluster_experiment_a_per_cluster.csv`
