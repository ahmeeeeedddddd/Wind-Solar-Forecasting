"""
arima_and_diagnostics.py - Time-Series Statistical Analysis & ARIMA Models

Includes missing time-series diagnostics from the lecture notebook:
  1. Stationarity Analysis (ADF & KPSS hypothesis tests)
  2. ACF & PACF plots for raw and differenced series
  3. Classical Statistical ARIMA forecasting (ARIMA(1,0,0), ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1), ARIMA(2,1,1))
  4. Residual diagnostics (Residual time series & Residual ACF plots)

Run from repo root:
    python src/arima_and_diagnostics.py

FIX (v2): in v1 the ARIMA models were fitted but the forecasts that were evaluated were
`full_series.shift(h)` (= the naive baseline), so every ARIMA row had identical MAE/RMSE/MASE.
Now each ARIMA model produces its OWN forecasts: parameters are fitted once on the last 4,000
training hours, then for forecast origins spread through the test period (every `stride` hours)
the fitted parameters are applied to the most recent `window` hours (no re-fitting, no future
data) and h-step-ahead forecasts are taken. Naive is evaluated on exactly the same origins so
the comparison is fair.
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
PRED_DIR   = os.path.join(BASE_DIR, "results", "predictions")


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
ORDERS = {
    "ARIMA(1,0,0)": (1, 0, 0),
    "ARIMA(1,1,0)": (1, 1, 0),
    "ARIMA(0,1,1)": (0, 1, 1),
    "ARIMA(1,1,1)": (1, 1, 1),
    "ARIMA(2,1,1)": (2, 1, 1),
}


def run_arima_models(full_series: pd.Series, train_end_idx, test_idx, scale: float,
                     stride: int = 6, window: int = 336):
    """
    Fit candidate ARIMA models and evaluate their REAL h-step-ahead forecasts on the test period.

    full_series : regular hourly Production series (gaps = NaN)
    stride      : forecast origins are every `stride` hours inside the test period
    window      : number of most recent hours the fitted model is applied to at each origin
    """
    print("--- 2. Fitting Classical ARIMA Models ---")
    max_h = max(HORIZONS)

    # For computational efficiency on 40,000+ hourly readings, fit on the recent 4,000 training hours (~6 months)
    recent_train = full_series.loc[:train_end_idx].iloc[-4000:]

    # forecast origins: the last observed hour is `t0`; we forecast t0+1h ... t0+24h
    last_origin = full_series.index[-1] - pd.Timedelta(hours=max_h)
    origins = [t for t in test_idx[::stride] if t <= last_origin]
    print(f"  {len(origins)} forecast origins in the test period (every {stride}h)")

    # ---- naive baseline on exactly the same origins (the fair reference)
    rows = []
    for h in HORIZONS:
        tgt = pd.DatetimeIndex(origins) + pd.Timedelta(hours=h)
        y_true = full_series.reindex(tgt).values
        naive = full_series.reindex(origins).values
        res = evaluate(y_true, naive, scale)
        res.update({"Model": "Naive (same origins)", "Horizon_h": h, "AIC": np.nan})
        rows.append(res)

    fits = {}
    for name, order in ORDERS.items():
        try:
            fit = ARIMA(recent_train, order=order).fit()
            fits[name] = fit
            aic = round(fit.aic, 1)

            preds = {h: [] for h in HORIZONS}
            for t0 in origins:
                hist = full_series.loc[t0 - pd.Timedelta(hours=window - 1): t0]
                fc = fit.apply(hist, refit=False).forecast(steps=max_h).values
                for h in HORIZONS:
                    preds[h].append(fc[h - 1])

            for h in HORIZONS:
                tgt = pd.DatetimeIndex(origins) + pd.Timedelta(hours=h)
                res = evaluate(full_series.reindex(tgt).values, np.array(preds[h]), scale)
                res.update({"Model": name, "Horizon_h": h, "AIC": aic})
                rows.append(res)
            print(f"  [OK] {name} fitted | AIC: {aic}")
        except Exception as e:
            print(f"  [Failed] {name}: {e}")

    df_arima = pd.DataFrame(rows)
    df_arima.to_csv(os.path.join(TABLES_DIR, "arima_results.csv"), index=False)

    show = df_arima.round({"MAE": 1, "RMSE": 1, "MASE": 4})
    print(show.to_string(index=False))
    return df_arima, fits


# --------------------------------------------------------------------------- 4. Residual Diagnostics
def plot_residual_diagnostics(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Best Model",
                              filename: str = "05_residual_diagnostics.png"):
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
    savefig(os.path.join(FIGS_DIR, filename))
    plt.close()
    print(f"Saved Residual Diagnostics plot -> results/figures/{filename}")


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
    
    # 3. ARIMA Models (real ARIMA forecasts, evaluated against naive on the same origins)
    df_arima, fits = run_arima_models(full, train.index.max(), test.index, scale)
    
    # 4. Residual Diagnostics
    y_test = full.reindex(test.index)

    # 4a. best ARIMA model at h=1 (one-step-ahead forecasts with the fitted parameters)
    arima_rows = df_arima[(df_arima.Horizon_h == 1) & (df_arima.Model.isin(fits.keys()))]
    if len(arima_rows):
        best_name = arima_rows.sort_values("MAE").iloc[0]["Model"]
        seg = full.loc[test.index.min() - pd.Timedelta(hours=336): test.index.max()]
        one_step = fits[best_name].apply(seg, refit=False).fittedvalues.reindex(test.index)
        plot_residual_diagnostics(y_test, one_step, model_name=f"{best_name} (h=1)",
                                  filename="05_residual_diagnostics.png")

    # 4b. naive baseline (for comparison)
    naive_pred = full.shift(1).reindex(test.index)
    plot_residual_diagnostics(y_test, naive_pred, model_name="Naive Baseline (h=1)",
                              filename="05b_residual_diagnostics_naive.png")

    # 4c. best supervised model (LightGBM) if models.py was already run
    p1 = os.path.join(PRED_DIR, "predictions_h1.csv")
    if os.path.exists(p1):
        pr = pd.read_csv(p1, index_col=0, parse_dates=True)
        if "LightGBM" in pr.columns:
            plot_residual_diagnostics(pr["y_true"], pr["LightGBM"], model_name="LightGBM (h=1)",
                                      filename="06_residual_diagnostics_lightgbm.png")
    
    print("\n=== All Time-Series Lecture Diagnostics Completed Successfully! ===")


if __name__ == "__main__":
    run_full_diagnostics()
