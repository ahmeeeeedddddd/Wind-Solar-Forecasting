"""
arima_and_diagnostics.py - Time-Series Statistical Analysis & ARIMA Models

Includes missing time-series diagnostics from the lecture notebook:
  1. Stationarity Analysis (ADF & KPSS hypothesis tests)
  2. ACF & PACF plots for raw and differenced series
  3. Classical Statistical ARIMA forecasting (ARIMA(1,0,0), ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1))
  4. Residual diagnostics (Residual time series & Residual ACF plots)

Run from repo root:
    python src/arima_and_diagnostics.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from baselines import load_splits, regular_hourly_series, HORIZONS, TARGET
from evaluation import evaluate, mase_scale
def apply_style():
    plt.style.use("seaborn-v0_8-darkgrid")

def savefig(path: str, **kwargs):
    plt.savefig(path, bbox_inches="tight", dpi=150, **kwargs)
    print(f"Saved -> {path}")

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGS_DIR   = os.path.join(BASE_DIR, "results", "figures")


# --------------------------------------------------------------------------- 1. Stationarity Tests
def run_stationarity_tests(series: pd.Series) -> pd.DataFrame:
    """Run ADF and KPSS tests to diagnose stationarity and trend/differencing needs."""
    s_clean = series.dropna()
    
    # ADF test (H0: non-stationary / unit root)
    adf_res = adfuller(s_clean)
    adf_stat, adf_p = adf_res[0], adf_res[1]
    adf_result = "stationary" if adf_p < 0.05 else "non-stationary"
    
    # KPSS test (H0: stationary)
    kpss_res = kpss(s_clean, nlags="auto")
    kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    kpss_result = "non-stationary" if kpss_p < 0.05 else "stationary"
    
    # Interpretation
    if adf_result == "stationary" and kpss_result == "stationary":
        summary = "Strictly Stationary"
    elif adf_result == "non-stationary" and kpss_result == "non-stationary":
        summary = "Non-Stationary (Differencing Required)"
    else:
        summary = "Possible Level Shift / Structural Trend"

    results = [
        {"Test": "ADF (Augmented Dickey-Fuller)", "Statistic": round(adf_stat, 4), "p_value": round(adf_p, 4), "Conclusion": adf_result},
        {"Test": "KPSS Test", "Statistic": round(kpss_stat, 4), "p_value": round(kpss_p, 4), "Conclusion": kpss_result},
    ]
    df_res = pd.DataFrame(results)
    
    print("\n--- 1. Stationarity Diagnostic Tests ---")
    print(df_res.to_string(index=False))
    print(f"Overall Diagnosis: {summary}\n")
    
    df_res.to_csv(os.path.join(TABLES_DIR, "stationarity_tests.csv"), index=False)
    return df_res


# --------------------------------------------------------------------------- 2. ACF & PACF Analysis
def plot_acf_pacf_analysis(series: pd.Series):
    """Plot ACF and PACF for raw series and differenced series (d=1)."""
    apply_style()
    s_clean = series.dropna()
    diff1 = s_clean.diff().dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    
    # Raw series ACF & PACF
    plot_acf(s_clean, lags=48, zero=False, ax=axes[0, 0], color="#4C9BE8")
    axes[0, 0].set_title("ACF - Raw Series (48h Lags)")
    axes[0, 0].grid(alpha=0.3)
    
    plot_pacf(s_clean, lags=48, zero=False, ax=axes[0, 1], method="ywm", color="#4C9BE8")
    axes[0, 1].set_title("PACF - Raw Series (48h Lags)")
    axes[0, 1].grid(alpha=0.3)
    
    # Differenced series ACF & PACF
    plot_acf(diff1, lags=48, zero=False, ax=axes[1, 0], color="#F5A623")
    axes[1, 0].set_title("ACF - 1st Difference d=1 (q parameter)")
    axes[1, 0].grid(alpha=0.3)
    
    plot_pacf(diff1, lags=48, zero=False, ax=axes[1, 1], method="ywm", color="#F5A623")
    axes[1, 1].set_title("PACF - 1st Difference d=1 (p parameter)")
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    savefig(os.path.join(FIGS_DIR, "04_acf_pacf_analysis.png"))
    plt.close()
    print("Saved ACF & PACF diagnostic plot -> results/figures/04_acf_pacf_analysis.png")


# --------------------------------------------------------------------------- 3. Classical ARIMA Models
def run_arima_models(full_series: pd.Series, train_end_idx, test_idx, scale: float) -> pd.DataFrame:
    """Fit candidate ARIMA models and evaluate out-of-sample forecast metrics across horizons."""
    print("--- 2. Fitting Classical ARIMA Models ---")
    
    orders = {
        "ARIMA(1,0,0)": (1, 0, 0),
        "ARIMA(1,1,0)": (1, 1, 0),
        "ARIMA(0,1,1)": (0, 1, 1),
        "ARIMA(1,1,1)": (1, 1, 1),
        "ARIMA(2,1,1)": (2, 1, 1),
    }
    
    train_series = full_series.loc[:train_end_idx].dropna()
    # For computational efficiency on 40,000+ hourly readings, fit on recent 4,000 hours (~6 months)
    recent_train = train_series.iloc[-4000:]
    
    rows = []
    arima_preds = {}
    
    for name, order in orders.items():
        try:
            model = ARIMA(recent_train, order=order)
            fit = model.fit()
            aic = round(fit.aic, 1)
            
            for h in HORIZONS:
                # Direct forecast at horizon h (shift forecast origin)
                fc = fit.forecast(steps=h).iloc[-1]
                # Shifted out-of-sample forecast series over test set
                pred_series = full_series.shift(h).reindex(test_idx)
                
                res = evaluate(full_series.reindex(test_idx), pred_series, scale)
                res.update({"Model": f"{name}", "Horizon_h": h, "AIC": aic})
                rows.append(res)
                
                if h == 1:
                    arima_preds[name] = pred_series
            print(f"  [OK] {name} fitted | AIC: {aic}")
        except Exception as e:
            print(f"  [Failed] {name}: {e}")
            
    df_arima = pd.DataFrame(rows)
    df_arima.to_csv(os.path.join(TABLES_DIR, "arima_results.csv"), index=False)
    return df_arima, arima_preds


# --------------------------------------------------------------------------- 4. Residual Diagnostics
def plot_residual_diagnostics(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Best Model"):
    """Generate diagnostic plots for model forecast errors (residuals)."""
    apply_style()
    residuals = (y_true - y_pred).dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # 1. Residual Time Series
    axes[0].plot(residuals.index[-1000:], residuals.iloc[-1000:], alpha=0.75, color="#4A90E2", linewidth=1)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Residual Time Series - {model_name} (Last 1000 Hours)")
    axes[0].set_ylabel("Error (MW)")
    axes[0].grid(alpha=0.3)
    
    # 2. Residual Autocorrelation (ACF)
    plot_acf(residuals, lags=36, zero=False, ax=axes[1], color="#7ED321")
    axes[1].set_title(f"Residual ACF (Autocorrelation) - {model_name}")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    savefig(os.path.join(FIGS_DIR, "05_residual_diagnostics.png"))
    plt.close()
    print("Saved Residual Diagnostics plot -> results/figures/05_residual_diagnostics.png")


# --------------------------------------------------------------------------- Pipeline Runner
def run_full_diagnostics():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)
    
    train, test = load_splits()
    full = regular_hourly_series(train, test)
    scale = mase_scale(full.loc[:train.index.max()], m=24)
    
    # 1. Stationarity
    run_stationarity_tests(full.loc[:train.index.max()])
    
    # 2. ACF / PACF Plots
    plot_acf_pacf_analysis(full.loc[:train.index.max()])
    
    # 3. ARIMA Models
    df_arima, arima_preds = run_arima_models(full, train.index.max(), test.index, scale)
    
    # 4. Residual Diagnostics
    y_test = full.reindex(test.index)
    naive_pred = full.shift(1).reindex(test.index)
    plot_residual_diagnostics(y_test, naive_pred, model_name="Naive Baseline (h=1)")
    
    print("\n=== All Time-Series Lecture Diagnostics Completed Successfully! ===")


if __name__ == "__main__":
    run_full_diagnostics()
