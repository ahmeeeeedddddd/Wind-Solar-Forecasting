# Person C Technical Report: Supervised Models & Evaluation

## Executive Summary
This report covers **Person C: supervised forecasting models and evaluation** on the France Wind & Solar Energy Production dataset. We forecast hourly production (`Production`, MW) at horizons of **1h, 6h and 24h** and compare three families of models on the same chronological test set:

1. **Baselines**: naive, seasonal naive (same hour yesterday), seasonal naive (same hour last week).
2. **Global supervised models**: Linear Regression, Ridge, Random Forest and LightGBM, tuned with `TimeSeriesSplit`.
3. **Cluster-aware models** built on Person B's daily-profile clusters.

Main findings:
- Every tuned machine-learning model beats every baseline at every horizon. The best honest (leak-free) model, **LightGBM**, reduces MAE by **29.3% (1h), 36.2% (6h) and 28.2% (24h)** relative to the best baseline.
- Forecast error grows quickly with the horizon: LightGBM MAE is 480 MW at 1h, 1,674 MW at 6h and 2,184 MW at 24h.
- Weather features matter more the further ahead we forecast (removing them raises LightGBM MAE by 1.4% at 1h, 7.7% at 6h and 25.6% at 24h).
- Errors differ strongly by season and cluster: at 24h, winter is the hardest season for LightGBM (MAE 2,621 MW) and summer the easiest (1,643 MW), and cluster 1 (high-production days) is the hardest cluster at every horizon.
- The apparent gains of cluster-aware models come almost entirely from **label leakage**. Person B's cluster label describes a whole day, including hours that are still in the future at forecast time. When we replace it with a label that is truly known at forecast time, the gain shrinks to 0.2-0.4% (within noise).

---

## 1. Supervised Methodology

### 1.1 Problem Formulation
Let $y_t$ be the production at hour $t$. For a horizon $h$, the forecast for target hour $T$ is made at the **forecast origin** $T-h$, using only information available at that moment. We use the **direct strategy**: a separate model is trained for each horizon $h \in \{1, 6, 24\}$, rather than feeding one-step predictions back into the model.

### 1.2 Data and Split
We reuse Person A's chronological 80/20 split (no shuffling):

| Item | Value |
|---|---|
| Training period | up to 2024-09-26 04:00 |
| Test period | 2024-09-26 05:00 to 2025-11-30 23:00 (about 10,300 hourly rows, roughly 14 months) |
| Train rows used (1h / 6h / 24h) | 41,169 / 41,169 / 41,174 |
| Test rows used (1h / 6h / 24h) | 10,335 / 10,335 / 10,336 |

The row counts are slightly below Person A's files (41,352 train / 10,338 test) because rows that lack a complete feature set (for example hours near gaps in the hourly series) are dropped. **All models and baselines at a given horizon are evaluated on exactly the same test rows.**

The raw series has a few missing hours. We therefore put the data on a strict hourly grid before building lags, so "24 hours earlier" always means 24 hours and not 24 rows.

### 1.3 Feature Engineering per Horizon
Person A's lag and rolling features (`lag_1h`, rolling statistics up to $t-1$) are valid only for a 1h forecast. Using them for 6h or 24h would leak information the forecaster would not have. We rebuilt the features for each horizon so that every production-based feature only looks at data up to $T-h$:

| Feature group | Features | Available at origin $T-h$? |
|---|---|---|
| Calendar of target hour | `hour`, `day_of_week`, `month`, `is_weekend` | Yes (known in advance) |
| Weather at target hour | `wind_speed_10m`, `temp_2m`, `irradiation`, `cloud_cover` | Assumed yes (perfect weather forecast, see Section 4) |
| Last known production | `last_known` = $y_{T-h}$ | Yes |
| Seasonal lags | `lag_24h` = $y_{T-24}$ (omitted at $h=24$, where it equals `last_known`), `lag_168h` = $y_{T-168}$ | Yes |
| Rolling statistics | mean and std of $y$ over 24h and 168h windows ending at $T-h$ | Yes |

This gives 15 features for 1h and 6h, and 14 for 24h.

### 1.4 Baselines
| Baseline | Forecast for $T$ (made at $T-h$) |
|---|---|
| Naive (last known value) | $y_{T-h}$ |
| Seasonal naive, same hour yesterday | $y_{T-24}$ |
| Seasonal naive, same hour last week | $y_{T-168}$ |

At $h=24$ the naive baseline and the daily seasonal naive baseline are identical by construction.

### 1.5 Models and Hyperparameter Tuning
| Model | Notes | Search |
|---|---|---|
| Linear Regression | Standardized inputs, no tuning | none |
| Ridge | Standardized inputs, `alpha` in {0.01, 0.1, 1, 10, 100, 1000} | grid search |
| Random Forest | `n_estimators` {100, 200}, `max_depth` {8, 12, 16, None}, `min_samples_leaf` {1, 5, 20}, `max_features` {0.5, 0.8, 1.0} | randomized search, 6 draws |
| LightGBM | `n_estimators` {200, 400, 800}, `learning_rate` {0.03, 0.05, 0.1}, `num_leaves` {15, 31, 63}, `min_child_samples` {20, 50, 100}, `subsample` {0.8, 1.0}, `colsample_bytree` {0.7, 1.0} | randomized search, 12 draws |

Tuning uses **`TimeSeriesSplit` with 3 expanding-window folds** on the training set only, scored by MAE. Each fold trains on the past and validates on the block that follows it, so the search never sees the future of the data it is validated on. Hyperparameters are tuned **once per horizon on the global (no-cluster) feature set**, then refit on the whole training set. The random seed is 42.

Selected hyperparameters and cross-validation MAE (also stored in `results/tables/best_hyperparameters.csv`):

| Horizon | Model | CV MAE (MW) | Selected parameters |
|---|---|---|---|
| 1h | Ridge | 453.3 | alpha = 100 |
| 1h | Random Forest | 379.8 | 100 trees, depth 16, min leaf 5, max features 1.0 |
| 1h | LightGBM | 351.7 | 800 trees, lr 0.03, 63 leaves, min child 20, subsample 1.0, colsample 1.0 |
| 6h | Ridge | 1,477.1 | alpha = 100 |
| 6h | Random Forest | 1,347.7 | 100 trees, depth None, min leaf 5, max features 0.8 |
| 6h | LightGBM | 1,316.8 | 200 trees, lr 0.05, 31 leaves, min child 20, subsample 1.0, colsample 1.0 |
| 24h | Ridge | 1,926.5 | alpha = 0.01 (lowest value in the grid) |
| 24h | Random Forest | 1,840.6 | 100 trees, depth 12, min leaf 20, max features 0.8 |
| 24h | LightGBM | 1,844.3 | 200 trees, lr 0.03, 31 leaves, min child 20, subsample 0.8, colsample 1.0 |

### 1.6 Cluster-Aware Models
We use Person B's K-Means ($k=3$) daily-profile labels and test three ways of using them, all with LightGBM and the tuned global parameters (no re-tuning):

- **(a) One model per cluster**: three LightGBM models, one per cluster; each test hour is routed to the model of its **same-day** label.
- **(b) Cluster label as a feature** using Person B's **same-day** label.
- **(b, leak-free) Cluster label as a feature** using the label of the day **D-2** (two days before the target day). A day that is two days back is always completely finished at the forecast origin for every horizon up to 24h, so this label is genuinely known at forecast time.

### 1.7 Evaluation Metrics
$$\text{MAE} = \frac{1}{n}\sum_{i}|y_i-\hat y_i| \qquad \text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i-\hat y_i)^2}$$

$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{N-24}\sum_{t=25}^{N}|y_t - y_{t-24}|}$$

The MASE denominator is the MAE of the 24h seasonal naive forecast on the **training** set, which equals **2,498.5 MW**. This is the same definition Person B used, so MASE values are comparable across the project. A MASE below 1 means the model is better than "same hour yesterday" measured on the training data. We use the same denominator for all horizons.

Errors are also reported by **meteorological season** (Winter = Dec-Feb, Spring = Mar-May, Summer = Jun-Aug, Autumn = Sep-Nov) and by **Person B's daily cluster**.

---

## 2. Experimental Setup

| Item | Value |
|---|---|
| Hardware | Intel Core i5, 16 GB RAM |
| Software | Python 3.14, scikit-learn 1.9.1, LightGBM 4.7.0, pandas, NumPy |
| Code | `src/evaluation.py` (metrics), `src/baselines.py` (baselines), `src/models.py` (features, tuning, models, comparison) |
| Command | `python src/models.py` (about 2.9 minutes for the full run) |
| Outputs | `results/tables/final_comparison.csv`, `final_by_season.csv`, `final_by_cluster.csv`, `best_hyperparameters.csv`, and per-horizon predictions in `results/predictions/` |
| Randomness | Seed 42 for all models and searches |

---

## 3. Results and Comparison

### 3.1 Overall Comparison (test set, sorted by MAE)

**Horizon 1h**

| Group | Model | MAE (MW) | RMSE (MW) | MASE |
|---|---|---|---|---|
| Cluster-aware | LightGBM + cluster feature (b), same-day label* | 469.19 | 836.19 | 0.1878 |
| Cluster-aware | LightGBM + cluster feature (b, leak-free) | 479.14 | 849.16 | 0.1918 |
| Global | LightGBM | 480.19 | 849.57 | 0.1922 |
| Global | LightGBM (no weather) | 486.87 | 860.96 | 0.1949 |
| Cluster-aware | LightGBM + per-cluster models (a), same-day label* | 488.75 | 838.90 | 0.1956 |
| Global | Random Forest | 503.98 | 865.88 | 0.2017 |
| Global | Linear Regression | 600.88 | 933.08 | 0.2405 |
| Global | Ridge | 601.78 | 932.48 | 0.2409 |
| Baseline | Naive (last known value) | 678.82 | 1,025.01 | 0.2717 |
| Baseline | Seasonal naive (same hour yesterday) | 3,043.59 | 4,111.30 | 1.2182 |
| Baseline | Seasonal naive (same hour last week) | 4,498.37 | 5,976.79 | 1.8004 |

**Horizon 6h**

| Group | Model | MAE (MW) | RMSE (MW) | MASE |
|---|---|---|---|---|
| Cluster-aware | LightGBM + cluster feature (b), same-day label* | 1,466.25 | 1,928.10 | 0.5869 |
| Cluster-aware | LightGBM + per-cluster models (a), same-day label* | 1,476.19 | 1,945.73 | 0.5908 |
| Cluster-aware | LightGBM + cluster feature (b, leak-free) | 1,670.35 | 2,180.62 | 0.6685 |
| Global | LightGBM | 1,673.54 | 2,179.75 | 0.6698 |
| Global | Random Forest | 1,700.19 | 2,216.70 | 0.6805 |
| Global | LightGBM (no weather) | 1,801.93 | 2,359.60 | 0.7212 |
| Global | Ridge | 1,838.49 | 2,360.24 | 0.7358 |
| Global | Linear Regression | 1,841.99 | 2,363.94 | 0.7372 |
| Baseline | Naive (last known value) | 2,621.30 | 3,432.45 | 1.0492 |
| Baseline | Seasonal naive (same hour yesterday) | 3,043.59 | 4,111.30 | 1.2182 |
| Baseline | Seasonal naive (same hour last week) | 4,498.55 | 5,976.87 | 1.8005 |

**Horizon 24h**

| Group | Model | MAE (MW) | RMSE (MW) | MASE |
|---|---|---|---|---|
| Cluster-aware | LightGBM + per-cluster models (a), same-day label* | 1,674.12 | 2,175.81 | 0.6701 |
| Cluster-aware | LightGBM + cluster feature (b), same-day label* | 1,677.84 | 2,170.99 | 0.6715 |
| Cluster-aware | LightGBM + cluster feature (b, leak-free) | 2,176.24 | 2,868.16 | 0.8710 |
| Global | LightGBM | 2,184.39 | 2,876.02 | 0.8743 |
| Global | Random Forest | 2,219.11 | 2,918.49 | 0.8882 |
| Global | Linear Regression | 2,294.56 | 2,999.71 | 0.9184 |
| Global | Ridge | 2,294.56 | 2,999.71 | 0.9184 |
| Global | LightGBM (no weather) | 2,743.55 | 3,577.75 | 1.0981 |
| Baseline | Naive (last known value) | 3,043.54 | 4,111.17 | 1.2181 |
| Baseline | Seasonal naive (same hour yesterday) | 3,043.54 | 4,111.17 | 1.2181 |
| Baseline | Seasonal naive (same hour last week) | 4,498.28 | 5,976.60 | 1.8004 |

\* These variants use Person B's same-day label, which is computed from the whole day and is therefore **not available at forecast time** (see Section 3.3). Their numbers are an optimistic upper bound, not a deployable result.

### 3.2 Summary Across Horizons (MAE, MW)

| Model | 1h | 6h | 24h |
|---|---|---|---|
| Best baseline | 678.82 (naive) | 2,621.30 (naive) | 3,043.54 (naive / seasonal naive) |
| Linear Regression | 600.88 | 1,841.99 | 2,294.56 |
| Random Forest | 503.98 | 1,700.19 | 2,219.11 |
| **LightGBM (global)** | **480.19** | **1,673.54** | **2,184.39** |
| LightGBM + cluster feature (b, leak-free) | 479.14 | 1,670.35 | 2,176.24 |
| LightGBM + cluster feature (b, same-day label)* | 469.19 | 1,466.25 | 1,677.84 |
| LightGBM improvement over best baseline | -29.3% | -36.2% | -28.2% |

### 3.3 Cluster-Aware vs Global: the Leakage Effect
Person B's label for a day is derived from all 24 hourly values of that day. When we forecast hour 8:00, a same-day label already "knows" whether the afternoon will be a high-production day. It is like solving an exam with a summary of the answers in your hand. The further ahead we forecast, the less recent production the model sees, so the more it leans on this hidden information:

| Horizon | Global LightGBM | Cluster feature (b, same-day label) | Change | Cluster feature (b, leak-free) | Change |
|---|---|---|---|---|---|
| 1h | 480.19 | 469.19 | -2.3% | 479.14 | -0.2% |
| 6h | 1,673.54 | 1,466.25 | -12.4% | 1,670.35 | -0.2% |
| 24h | 2,184.39 | 1,677.84 | -23.2% | 2,176.24 | -0.4% |

The same-day label produces large "improvements" that grow with the horizon (up to 23.2% at 24h). With a label that is actually known at forecast time, the improvement is only 0.2-0.4%. We ran a single chronological split and did not test for significance, but a difference this small (1 to 8 MW against errors of 480 to 2,184 MW) should be treated as no practical gain.

Experiment (a) uses the same same-day routing, so it is affected by the same leakage. At 1h it is even worse than the global model (488.75 vs 480.19), which is consistent with Person B's observation that splitting the training data across clusters leaves each model with fewer samples.

### 3.4 Other Findings
1. **Machine learning beats the baselines at every horizon.** At 1h the naive forecast is a strong baseline (MASE 0.27) because production changes slowly from one hour to the next. The seasonal naive baselines are poor (MASE above 1) because weather changes from day to day and week to week. At 6h and 24h the best baseline is itself worse than yesterday's pattern measured on the training set (MASE above 1).
2. **Non-linear models win.** LightGBM beats Linear Regression by 20.1% at 1h and 4.8% at 24h. Production depends non-linearly on its inputs (the solar bell curve, wind power rising steeply with wind speed). Ridge is practically identical to ordinary Linear Regression: the selected `alpha` is 100 at 1h and 6h, and 0.01 (the lowest value in the grid) at 24h, where the two models give exactly the same MAE. Regularization adds nothing with 14-15 features and 41,000 training rows.
3. **LightGBM is slightly better than Random Forest** (by 4.7% at 1h and about 1.6% at 6h and 24h) and trains much faster.
4. **Weather becomes essential for longer horizons.** Removing the weather features raises LightGBM MAE by 1.4% at 1h, 7.7% at 6h and 25.6% at 24h. At 1h the last known production already tells most of the story; at 24h the model must rely on weather.
5. **Error grows with the horizon.** LightGBM MAE rises from 480 MW (1h) to 1,674 MW (6h) and 2,184 MW (24h). Most of the loss happens between 1h and 6h; from 6h to 24h the increase is smaller.
6. **Consistency check with Person B.** Our tuned Random Forest at 1h (MAE 503.98) is close to Person B's untuned Random Forest (509.88). The small difference comes from slightly different features, row filtering and tuning.
7. **The test period is harder than the training period.** The daily seasonal naive MASE on the test set is 1.22, whereas by definition it is about 1.0 on the training data. This also explains why cross-validation MAE (for example LightGBM at 1h: 351.7 MW) is lower than test MAE (480.2 MW).

### 3.5 Error by Season and by Cluster
All numbers are MAE in MW on the test set. Full tables with RMSE and MASE for every model are in `results/tables/final_by_season.csv` and `results/tables/final_by_cluster.csv`. Rows marked * use the same-day cluster label and are therefore optimistic (Section 3.3).

#### 3.5.1 By Season
The test period runs from 2024-09-26 to 2025-11-30, so it contains one winter (2160 hours), one spring (2204), one summer (2208) and two autumns (3763 hours, partly 2024 and partly 2025). Each seasonal result therefore rests on a single year, and autumn is better represented than the others.

**Horizon 1h**

| Model | Winter | Spring | Summer | Autumn |
|---|---|---|---|---|
| Best baseline (lowest MAE of the three) | 486.2 | 765.2 | 828.0 | 651.3 |
| Linear Regression | 478.5 | 684.2 | 662.4 | 586.2 |
| Random Forest | 411.3 | 589.4 | 520.5 | 497.5 |
| LightGBM (global) | 403.4 | 563.0 | 486.6 | 472.0 |
| LightGBM (no weather) | 406.3 | 578.2 | 490.3 | 477.6 |
| LightGBM + cluster feature (b, leak-free) | 401.6 | 558.7 | 487.6 | 472.1 |
| LightGBM + cluster feature (b), same-day label* | 383.7 | 550.6 | 485.4 | 461.0 |
| LightGBM + per-cluster models (a), same-day label* | 391.9 | 572.5 | 510.1 | 482.8 |

**Horizon 6h**

| Model | Winter | Spring | Summer | Autumn |
|---|---|---|---|---|
| Best baseline (lowest MAE of the three) | 2,094.4 | 2,481.4 | 2,230.0 | 2,612.7 |
| Linear Regression | 1,793.2 | 1,840.9 | 1,824.2 | 1,881.1 |
| Random Forest | 1,698.8 | 1,771.5 | 1,565.1 | 1,738.5 |
| LightGBM (global) | 1,670.2 | 1,771.4 | 1,542.9 | 1,694.8 |
| LightGBM (no weather) | 1,843.2 | 1,811.5 | 1,712.1 | 1,825.3 |
| LightGBM + cluster feature (b, leak-free) | 1,670.2 | 1,769.8 | 1,540.6 | 1,688.4 |
| LightGBM + cluster feature (b), same-day label* | 1,458.7 | 1,516.2 | 1,375.4 | 1,494.6 |
| LightGBM + per-cluster models (a), same-day label* | 1,457.2 | 1,539.1 | 1,375.0 | 1,509.6 |

**Horizon 24h**

| Model | Winter | Spring | Summer | Autumn |
|---|---|---|---|---|
| Best baseline (lowest MAE of the three) | 3,693.5 | 2,697.6 | 2,230.0 | 3,350.5 |
| Linear Regression | 2,856.1 | 2,111.2 | 1,743.5 | 2,403.0 |
| Random Forest | 2,706.9 | 2,122.3 | 1,651.8 | 2,328.7 |
| LightGBM (global) | 2,620.6 | 2,081.7 | 1,642.9 | 2,311.9 |
| LightGBM (no weather) | 3,673.9 | 2,331.7 | 1,926.6 | 2,930.2 |
| LightGBM + cluster feature (b, leak-free) | 2,600.7 | 2,075.1 | 1,638.0 | 2,307.7 |
| LightGBM + cluster feature (b), same-day label* | 1,895.0 | 1,690.8 | 1,447.0 | 1,681.0 |
| LightGBM + per-cluster models (a), same-day label* | 1,871.0 | 1,721.6 | 1,438.8 | 1,671.4 |

Findings by season:
1. **The hardest season depends on the horizon.** For LightGBM, the lowest error at 1h is in winter (403 MW) and the highest in spring (563 MW). At 6h, summer is the easiest (1,543 MW) and spring the hardest (1,771 MW). At 24h, summer is again the easiest (1,643 MW), but **winter becomes by far the hardest (2,621 MW, MASE 1.049)**.
2. **Persistence works in summer but fails in winter.** Summer production follows a regular daily solar cycle, so "same hour yesterday" has MASE 0.89 in summer and 1.48 in winter. The weekly seasonal naive baseline is especially poor in winter (MAE 7,252 MW, MASE 2.90), which is expected because weather regimes change from week to week.
3. **LightGBM beats the best baseline in every season at 24h**, by 31.0% in autumn, 29.0% in winter, 26.3% in summer and 22.8% in spring.
4. **Weather matters most in winter.** At 24h, removing the weather features raises LightGBM MAE by 40.2% in winter, 26.7% in autumn, 17.3% in summer and 12.0% in spring. This is consistent with winter production being driven by wind, which cannot be predicted from yesterday's production alone (this is our interpretation and was not tested separately for wind and solar).
5. **The leakage effect appears in every season.** At 24h the same-day label lowers LightGBM MAE by 27.7% in winter, 27.3% in autumn, 18.8% in spring and 11.9% in summer, while the leak-free label changes it by only 0.2% to 0.8%.

#### 3.5.2 By Cluster
Clusters are Person B's K-Means daily-profile clusters. In the test set, cluster 0 has 4080 hours (39.5%), cluster 1 has about 2033 hours (19.7%) and cluster 2 has 4222 hours (40.9%). The grouping uses the same-day label, which is fine for describing where errors occur but is not a forecasting input here. In Person B's naming, cluster 0 is the low-production group, cluster 1 the high-production group and cluster 2 the moderate group (please confirm against `results/tables/cluster_summary.csv`).

**Horizon 1h**

| Model | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Best baseline (lowest MAE of the three) | 570.8 | 746.3 | 750.7 |
| Linear Regression | 496.3 | 708.2 | 650.3 |
| Random Forest | 378.2 | 655.2 | 552.7 |
| LightGBM (global) | 351.1 | 679.8 | 508.8 |
| LightGBM (no weather) | 358.1 | 672.1 | 522.1 |
| LightGBM + cluster feature (b, leak-free) | 349.8 | 679.7 | 507.6 |
| LightGBM + cluster feature (b), same-day label* | 341.2 | 657.9 | 502.0 |
| LightGBM + per-cluster models (a), same-day label* | 351.0 | 668.2 | 535.4 |

**Horizon 6h**

| Model | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Best baseline (lowest MAE of the three) | 2,186.7 | 2,930.6 | 2,837.7 |
| Linear Regression | 1,397.2 | 2,540.6 | 1,935.5 |
| Random Forest | 1,175.3 | 2,488.2 | 1,828.0 |
| LightGBM (global) | 1,184.8 | 2,420.6 | 1,786.2 |
| LightGBM (no weather) | 1,335.8 | 2,600.6 | 1,867.8 |
| LightGBM + cluster feature (b, leak-free) | 1,175.3 | 2,432.3 | 1,781.9 |
| LightGBM + cluster feature (b), same-day label* | 1,028.8 | 2,110.8 | 1,578.7 |
| LightGBM + per-cluster models (a), same-day label* | 1,022.2 | 2,158.7 | 1,586.3 |

**Horizon 24h**

| Model | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Best baseline (lowest MAE of the three) | 2,186.7 | 4,763.8 | 3,042.8 |
| Linear Regression | 1,691.3 | 3,788.8 | 2,157.6 |
| Random Forest | 1,491.4 | 3,762.9 | 2,178.6 |
| LightGBM (global) | 1,480.0 | 3,709.2 | 2,130.5 |
| LightGBM (no weather) | 2,313.7 | 4,748.2 | 2,193.2 |
| LightGBM + cluster feature (b, leak-free) | 1,473.8 | 3,683.6 | 2,128.9 |
| LightGBM + cluster feature (b), same-day label* | 1,146.7 | 2,511.5 | 1,789.5 |
| LightGBM + per-cluster models (a), same-day label* | 1,142.3 | 2,476.0 | 1,801.8 |

Findings by cluster:
1. **Cluster 1 (high-production days) is the hardest to forecast**, and cluster 0 the easiest, at every horizon. LightGBM MAE is 351 / 509 / 680 MW for clusters 0 / 2 / 1 at 1h, and 1,480 / 2,131 / 3,709 MW at 24h. Part of this is simply scale: absolute errors in MW grow with the production level, and the MASE denominator is one global constant, so MASE does not remove that effect.
2. **At 24h, LightGBM on cluster 1 has MASE 1.485**, worse than the training-set seasonal naive scale, but it is still 22.1% better than the naive forecast on those same days (3,709 vs 4,764 MW).
3. **The same-day label helps the hardest cluster the most**: at 24h it lowers LightGBM MAE by 32.3% on cluster 1, 22.5% on cluster 0 and 16.0% on cluster 2. The leak-free label changes it by only 0.7%, 0.4% and 0.1%. The label effectively tells the model "today is a high-production day", which is exactly the information a real forecaster does not have.
4. **Weather dependence differs by cluster.** At 24h, removing weather raises MAE by 56.3% on cluster 0, 28.0% on cluster 1 and only 2.9% on cluster 2.
5. **Per-cluster models (a) do not help at 1h**: on cluster 2 the per-cluster model is 5.2% worse than the global model (535 vs 509 MW), consistent with having less training data per model. At 1h, Random Forest is slightly better than LightGBM on cluster 1 (655 vs 680 MW), although LightGBM is better overall.

---

## 4. Limitations
- **Perfect weather assumption.** The weather columns at the target hour are observed values, not forecasts. In real operation weather forecasts have errors, so the benefit of weather features, especially at 6h and 24h, is probably overstated.
- **Cluster labels were built on all days.** Person B's clustering was fitted on the whole dataset, including the test days. Even the leak-free D-2 label comes from that clustering, although the clustering is unsupervised and the effect is probably small.
- **Experiment (a) has no leak-free version.** Only the same-day routing was tested for per-cluster models.
- **Cluster-aware models reuse global hyperparameters.** They were not tuned separately, which may slightly favour the global model.
- **Single split.** Results come from one chronological split without confidence intervals or significance tests.
- **One year per season.** The test period contains a single winter, spring and summer (and two autumns), so seasonal conclusions may not generalize to other years.
- **Single aggregated target.** Production combines wind and solar, so we cannot say which source drives the errors.

## 5. Conclusion
Gradient boosting (LightGBM) with horizon-specific features is the best deployable model at all three horizons, improving on the best baseline by 28% to 36%. Weather information is the most important driver at longer horizons. Cluster-aware modelling gives no practical improvement once the cluster label is restricted to information available at forecast time; the large gains seen with the same-day label are an artifact of leakage. We recommend using the global LightGBM model, optionally with the leak-free cluster label, as the final forecasting model.

---

## Deliverable Artifacts Summary

- **Source Code**: `src/evaluation.py`, `src/baselines.py`, `src/models.py`
- **Tables**: `results/tables/final_comparison.csv`, `final_by_season.csv`, `final_by_cluster.csv`, `best_hyperparameters.csv`, `baselines_results.csv`, `baselines_by_season.csv`, `baselines_by_cluster.csv`
- **Predictions**: `results/predictions/predictions_h1.csv`, `predictions_h6.csv`, `predictions_h24.csv`
