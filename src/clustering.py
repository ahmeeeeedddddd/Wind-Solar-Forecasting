"""
clustering.py - Person B
Unsupervised learning and cluster-aware forecasting.
Builds daily profile matrix, evaluates clusters, fits KMeans/GMM/Hierarchical models,
visualizes results, and runs cluster-aware forecasting experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, mean_absolute_error, root_mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.cluster.hierarchy import dendrogram, linkage

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def build_daily_profile_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot hourly production data into a daily profile matrix (one row per date, 24 hourly columns).
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["date"] = df.index.date
    df["hour"] = df.index.hour

    # Pivot: index=date, columns=hour, values=Production
    pivot_df = df.pivot(index="date", columns="hour", values="Production")

    # Filter out days that don't have full 24 hours
    pivot_df = pivot_df.dropna(thresh=24)
    pivot_df.index = pd.to_datetime(pivot_df.index)
    return pivot_df


def normalize_profiles(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize daily profile matrix (StandardScaler per hourly feature across days).
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(pivot_df)
    scaled_df = pd.DataFrame(scaled_array, index=pivot_df.index, columns=pivot_df.columns)
    return scaled_df


def evaluate_k(scaled_df: pd.DataFrame, max_k: int = 10, save_plot: bool = True):
    """
    Evaluate elbow (inertia) and silhouette score for k in range 2..max_k.
    """
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(scaled_df)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(scaled_df, labels))

    if save_plot:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Inertia (Elbow)', color=color, fontsize=12, fontweight='bold')
        ax1.plot(k_range, inertias, 'o--', color=color, linewidth=2, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Silhouette Score', color=color, fontsize=12, fontweight='bold')
        ax2.plot(k_range, silhouettes, 's-', color=color, linewidth=2, label='Silhouette')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Clustering Diagnostics: Elbow Method & Silhouette Score', fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "elbow_silhouette.png"), dpi=300)
        plt.close()

    metrics_df = pd.DataFrame({
        "k": k_range,
        "inertia": inertias,
        "silhouette": silhouettes
    })
    metrics_df.to_csv(os.path.join(TABLES_DIR, "clustering_k_selection.csv"), index=False)
    return metrics_df


def fit_clustering_algorithms(scaled_df: pd.DataFrame, k: int = 3):
    """
    Fit KMeans, GaussianMixture, and AgglomerativeClustering for given k.
    """
    # 1. KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km_labels = km.fit_predict(scaled_df)

    # 2. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
    gmm_labels = gmm.fit_predict(scaled_df)

    # 3. Hierarchical Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    agg_labels = agg.fit_predict(scaled_df)

    labels_df = pd.DataFrame({
        "date": scaled_df.index,
        "kmeans": km_labels,
        "gmm": gmm_labels,
        "hierarchical": agg_labels
    }).set_index("date")

    # Evaluation comparison table
    summary = []
    for name, labels in [("KMeans", km_labels), ("GMM", gmm_labels), ("Hierarchical", agg_labels)]:
        sil = silhouette_score(scaled_df, labels)
        summary.append({"Algorithm": name, "Silhouette_Score": sil})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(TABLES_DIR, "clustering_algorithms_comparison.csv"), index=False)

    return labels_df, summary_df


def plot_dendrogram_tree(scaled_df: pd.DataFrame):
    """Plot hierarchical clustering dendrogram."""
    plt.figure(figsize=(12, 6))
    Z = linkage(scaled_df, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index / (Cluster Size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dendrogram.png"), dpi=300)
    plt.close()


def interpret_and_plot_clusters(raw_df: pd.DataFrame, labels_df: pd.DataFrame, algorithm: str = "kmeans"):
    """
    Calculate cluster centroids, plot daily average profiles, and map seasonal distributions.
    """
    df_merged = raw_df.copy()
    df_merged["cluster"] = labels_df[algorithm]

    # Cluster profiles (24h hourly average)
    profile_means = df_merged.groupby("cluster").mean()
    profile_stds = df_merged.groupby("cluster").std()

    # Name mapping based on average production profile characteristics
    # Cluster ordering: sort by daily total energy production
    totals = profile_means.sum(axis=1).sort_values(ascending=True)
    sorted_clusters = totals.index.tolist()

    # Give descriptive names
    cluster_names = {}
    if len(sorted_clusters) == 3:
        cluster_names[sorted_clusters[0]] = "Low Production (Winter / Overcast)"
        cluster_names[sorted_clusters[1]] = "Moderate Production (Transitional)"
        cluster_names[sorted_clusters[2]] = "High Production (Summer / Sunny)"
    else:
        for idx, c in enumerate(sorted_clusters):
            cluster_names[c] = f"Cluster {c} (Avg Daily: {totals[c]:.0f} MW)"

    df_merged["cluster_name"] = df_merged["cluster"].map(cluster_names)

    # Plot average 24-hour profiles
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("Set2", len(profile_means))

    hours = np.arange(24)
    for c in profile_means.index:
        mean_p = profile_means.loc[c]
        std_p = profile_stds.loc[c]
        name = cluster_names.get(c, f"Cluster {c}")
        plt.plot(hours, mean_p, marker='o', label=name, linewidth=2.5, color=palette[c])
        plt.fill_between(hours, mean_p - 0.5 * std_p, mean_p + 0.5 * std_p, alpha=0.15, color=palette[c])

    plt.xticks(hours)
    plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
    plt.ylabel('Energy Production (MW)', fontsize=12, fontweight='bold')
    plt.title('Daily Hourly Energy Production Profiles by Cluster (Mean ± 0.5 STD)', fontsize=14, fontweight='bold')
    plt.legend(title='Cluster Name', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cluster_profiles.png"), dpi=300)
    plt.close()

    # Seasonal breakdown
    df_merged["month"] = df_merged.index.month
    monthly_cluster = pd.crosstab(df_merged["month"], df_merged["cluster_name"], normalize='index') * 100

    plt.figure(figsize=(12, 6))
    monthly_cluster.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
    plt.xlabel('Month', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Days (%)', fontsize=12, fontweight='bold')
    plt.title('Monthly Distribution of Clusters', fontsize=14, fontweight='bold')
    plt.legend(title='Cluster Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cluster_monthly_distribution.png"), dpi=300)
    plt.close()

    # Save summary table
    summary_list = []
    for c in profile_means.index:
        days_count = (df_merged["cluster"] == c).sum()
        pct_days = days_count / len(df_merged) * 100
        mean_daily_prod = totals[c]
        peak_hour = profile_means.loc[c].idxmax()
        peak_val = profile_means.loc[c].max()
        summary_list.append({
            "Cluster": c,
            "Cluster_Name": cluster_names.get(c, f"Cluster {c}"),
            "Days_Count": days_count,
            "Pct_Days": round(pct_days, 2),
            "Mean_Daily_Total_MW": round(mean_daily_prod, 2),
            "Peak_Hour": peak_hour,
            "Peak_Production_MW": round(peak_val, 2)
        })

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(TABLES_DIR, "cluster_summary.csv"), index=False)

    return df_merged, cluster_names


def plot_pca_visualization(scaled_df: pd.DataFrame, labels_df: pd.DataFrame, algorithm: str = "kmeans"):
    """
    Apply PCA (2 components) and scatter plot days colored by cluster.
    """
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_df)

    pca_df = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "cluster": labels_df[algorithm].values
    }, index=scaled_df.index)

    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="cluster", palette="Set2",
        s=50, alpha=0.8, edgecolor="k", linewidth=0.5
    )

    plt.title(f'PCA Visualization of Daily Profiles (PC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%)',
              fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}% Variance)', fontsize=12)
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pca_clusters.png"), dpi=300)
    plt.close()

    return pca_df


def merge_cluster_labels_to_features(labels_df: pd.DataFrame):
    """
    Merge cluster labels onto train_features and test_features datasets.
    """
    train_feat = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"), index_col="datetime", parse_dates=True)
    test_feat = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features.csv"), index_col="datetime", parse_dates=True)

    # Add date column for joining
    train_feat["date"] = pd.to_datetime(train_feat.index.date)
    test_feat["date"] = pd.to_datetime(test_feat.index.date)

    # Make labels index datetime
    labels_copy = labels_df.copy()
    labels_copy.index = pd.to_datetime(labels_copy.index)

    # Primary cluster column is kmeans
    labels_to_merge = labels_copy[["kmeans"]].rename(columns={"kmeans": "cluster"})

    train_merged = train_feat.merge(labels_to_merge, left_on="date", right_index=True, how="left")
    test_merged = test_feat.merge(labels_to_merge, left_on="date", right_index=True, how="left")

    # If any missing cluster in test (e.g. incomplete day boundary), forward/bfill
    train_merged["cluster"] = train_merged["cluster"].ffill().bfill().astype(int)
    test_merged["cluster"] = test_merged["cluster"].ffill().bfill().astype(int)

    train_merged = train_merged.drop(columns=["date"])
    test_merged = test_merged.drop(columns=["date"])

    train_merged.to_csv(os.path.join(PROCESSED_DIR, "train_features_clustered.csv"))
    test_merged.to_csv(os.path.join(PROCESSED_DIR, "test_features_clustered.csv"))
    labels_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_labels.csv"))

    print("Cluster features merged and saved successfully.")
    return train_merged, test_merged


def calculate_mase(y_true, y_pred, y_train, seasonality=24):
    """Calculate Mean Absolute Scaled Error (MASE)."""
    mae = mean_absolute_error(y_true, y_pred)
    naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return mae / naive_mae if naive_mae != 0 else np.nan


def run_cluster_experiments():
    """
    Run Cluster-Aware Forecasting Experiments:
      (a) One forecasting model per cluster
      (b) Cluster label as an extra feature in a single global model
      Baseline: Global RF without cluster label
    """
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features_clustered.csv"), index_col="datetime", parse_dates=True)
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features_clustered.csv"), index_col="datetime", parse_dates=True)

    # Exclude non-feature columns
    ignore_cols = ["Source"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols + ["Production", "cluster"]]

    X_train_base = train_df[feature_cols]
    y_train = train_df["Production"]
    X_test_base = test_df[feature_cols]
    y_test = test_df["Production"]

    # -------------------------------------------------------------
    # Baseline: Global Random Forest Model (No Cluster Information)
    # -------------------------------------------------------------
    global_rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    global_rf.fit(X_train_base, y_train)
    preds_baseline = global_rf.predict(X_test_base)

    mae_base = mean_absolute_error(y_test, preds_baseline)
    rmse_base = root_mean_squared_error(y_test, preds_baseline)
    mase_base = calculate_mase(y_test.values, preds_baseline, y_train.values)

    print(f"Global RF Baseline -> MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}, MASE: {mase_base:.4f}")

    # -------------------------------------------------------------
    # Experiment (a): One Forecasting Model Per Cluster
    # -------------------------------------------------------------
    unique_clusters = sorted(train_df["cluster"].unique())
    preds_exp_a = pd.Series(index=test_df.index, dtype=float)
    cluster_models_a = {}
    exp_a_per_cluster_metrics = []

    for c in unique_clusters:
        train_c = train_df[train_df["cluster"] == c]
        test_c = test_df[test_df["cluster"] == c]

        if len(train_c) == 0:
            continue

        model_c = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model_c.fit(train_c[feature_cols], train_c["Production"])
        cluster_models_a[c] = model_c

        if len(test_c) > 0:
            preds_c = model_c.predict(test_c[feature_cols])
            preds_exp_a.loc[test_c.index] = preds_c

            mae_c = mean_absolute_error(test_c["Production"], preds_c)
            rmse_c = root_mean_squared_error(test_c["Production"], preds_c)
            exp_a_per_cluster_metrics.append({
                "Cluster": c,
                "Test_Samples": len(test_c),
                "MAE": round(mae_c, 2),
                "RMSE": round(rmse_c, 2)
            })

    mae_exp_a = mean_absolute_error(y_test, preds_exp_a)
    rmse_exp_a = root_mean_squared_error(y_test, preds_exp_a)
    mase_exp_a = calculate_mase(y_test.values, preds_exp_a.values, y_train.values)

    print(f"Experiment (a) Per-Cluster Models -> MAE: {mae_exp_a:.2f}, RMSE: {rmse_exp_a:.2f}, MASE: {mase_exp_a:.4f}")

    # -------------------------------------------------------------
    # Experiment (b): Cluster Label as Extra Feature in Global Model
    # -------------------------------------------------------------
    feature_cols_exp_b = feature_cols + ["cluster"]
    X_train_b = train_df[feature_cols_exp_b]
    X_test_b = test_df[feature_cols_exp_b]

    global_rf_b = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    global_rf_b.fit(X_train_b, y_train)
    preds_exp_b = global_rf_b.predict(X_test_b)

    mae_exp_b = mean_absolute_error(y_test, preds_exp_b)
    rmse_exp_b = root_mean_squared_error(y_test, preds_exp_b)
    mase_exp_b = calculate_mase(y_test.values, preds_exp_b, y_train.values)

    print(f"Experiment (b) Cluster Feature Model -> MAE: {mae_exp_b:.2f}, RMSE: {rmse_exp_b:.2f}, MASE: {mase_exp_b:.4f}")

    # Save summary results tables
    exp_summary = [
        {"Model_Strategy": "Baseline Global RF (No Cluster)", "MAE": round(mae_base, 2), "RMSE": round(rmse_base, 2), "MASE": round(mase_base, 4)},
        {"Model_Strategy": "Experiment (a): Separate Model per Cluster", "MAE": round(mae_exp_a, 2), "RMSE": round(rmse_exp_a, 2), "MASE": round(mase_exp_a, 4)},
        {"Model_Strategy": "Experiment (b): Cluster Label as Extra Feature", "MAE": round(mae_exp_b, 2), "RMSE": round(rmse_exp_b, 2), "MASE": round(mase_exp_b, 4)}
    ]

    exp_summary_df = pd.DataFrame(exp_summary)
    exp_summary_df.to_csv(os.path.join(TABLES_DIR, "cluster_aware_experiments_summary.csv"), index=False)

    exp_a_df = pd.DataFrame(exp_a_per_cluster_metrics)
    exp_a_df.to_csv(os.path.join(TABLES_DIR, "cluster_experiment_a_per_cluster.csv"), index=False)

    return exp_summary_df


def run_full_pipeline():
    """Execute complete Person B unsupervised learning & forecasting pipeline."""
    print("=== Step 1: Loading cleaned production data and building daily matrix ===")
    cleaned_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    df_cleaned = pd.read_csv(cleaned_path, index_col="datetime", parse_dates=True)

    pivot_raw = build_daily_profile_matrix(df_cleaned)
    pivot_scaled = normalize_profiles(pivot_raw)
    print(f"Daily profile matrix shape: {pivot_raw.shape} (Days x 24 Hours)")

    print("=== Step 2: Evaluating optimal cluster count k ===")
    metrics_k = evaluate_k(pivot_scaled, max_k=10, save_plot=True)

    print("=== Step 3: Fitting KMeans, GMM, and Hierarchical models ===")
    best_k = 3  # Chosen based on elbow and silhouette evaluation
    labels_df, algo_summary = fit_clustering_algorithms(pivot_scaled, k=best_k)

    print("=== Step 4: Generating dendrogram, cluster profiles & seasonal analysis ===")
    plot_dendrogram_tree(pivot_scaled)
    interpreted_df, cluster_names = interpret_and_plot_clusters(pivot_raw, labels_df, algorithm="kmeans")

    print("=== Step 5: Visualizing PCA clusters ===")
    pca_df = plot_pca_visualization(pivot_scaled, labels_df, algorithm="kmeans")

    print("=== Step 6: Saving cluster labels and merged feature sets ===")
    train_clustered, test_clustered = merge_cluster_labels_to_features(labels_df)

    print("=== Step 7: Running cluster-aware forecasting experiments (a) & (b) ===")
    results_summary = run_cluster_experiments()

    print("=== Person B Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    run_full_pipeline()
