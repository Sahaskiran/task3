"""
Predictive Analytics Using Historical Data
==========================================
Build a predictive model to forecast future trends using regression
and time-series models on historical datasets.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. GENERATE HISTORICAL DATA
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

dates   = pd.date_range(start="2021-01-01", periods=48, freq="MS")
t       = np.arange(48)
trend   = 200 + 6.5 * t
seasonal = 60 * np.sin(2 * np.pi * t / 12)
noise   = np.random.normal(0, 18, 48)
sales   = trend + seasonal + noise

df = pd.DataFrame({"date": dates, "sales": sales, "t": t})
df["month"] = df["date"].dt.month

print("=" * 60)
print("PREDICTIVE ANALYTICS — HISTORICAL SALES DATA")
print("=" * 60)
print(f"\nDataset : {len(df)} monthly records (2021–2024)")
print(f"Sales range : {df.sales.min():.1f} – {df.sales.max():.1f}")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
df["sin12"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos12"] = np.cos(2 * np.pi * df["month"] / 12)
df["sin6"]  = np.sin(2 * np.pi * df["month"] / 6)

features = ["t", "sin12", "cos12", "sin6"]
X = df[features].values
y = df["sales"].values

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT (last 6 months as held-out test)
# ─────────────────────────────────────────────────────────────
split = 42
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = df["date"].iloc[split:]

# ─────────────────────────────────────────────────────────────
# 4. MODEL 1 — LINEAR REGRESSION WITH SEASONAL FOURIER TERMS
# ─────────────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_full  = lr.predict(X)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)

print(f"\n--- Linear Regression + Fourier Seasonality ---")
print(f"  MAE  : {lr_mae:.2f}  |  RMSE : {lr_rmse:.2f}  |  R² : {lr_r2:.4f}")

# ─────────────────────────────────────────────────────────────
# 5. MODEL 2 — RIDGE REGRESSION (regularized)
# ─────────────────────────────────────────────────────────────
rr = Ridge(alpha=1.0)
rr.fit(X_train, y_train)
rr_pred = rr.predict(X_test)

rr_mae  = mean_absolute_error(y_test, rr_pred)
rr_rmse = np.sqrt(mean_squared_error(y_test, rr_pred))
rr_r2   = r2_score(y_test, rr_pred)

print(f"\n--- Ridge Regression (α=1) ---")
print(f"  MAE  : {rr_mae:.2f}  |  RMSE : {rr_rmse:.2f}  |  R² : {rr_r2:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. FUTURE FORECAST — 6 months ahead (Jan–Jun 2025)
# ─────────────────────────────────────────────────────────────
future_t  = np.arange(48, 54)
future_dates = pd.date_range("2025-01-01", periods=6, freq="MS")
future_month = future_dates.month

X_future = np.column_stack([
    future_t,
    np.sin(2 * np.pi * future_month / 12),
    np.cos(2 * np.pi * future_month / 12),
    np.sin(2 * np.pi * future_month / 6),
])

forecast = lr.predict(X_future)
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast.round(1)})

print("\n--- 6-Month Forecast (Jan–Jun 2025) ---")
print(forecast_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Predictive Analytics — Sales Forecasting with Regression Models",
             fontsize=13, fontweight="bold")

# Plot 1 — historical data
ax = axes[0, 0]
ax.plot(df["date"], df["sales"], color="#1976D2", lw=1.8, label="Actual Sales")
ax.plot(df["date"], lr_full, color="#E53935", lw=1.4, linestyle="--", label="LR Fit")
ax.set_title("Historical Data & Model Fit (2021–2024)")
ax.set_ylabel("Sales"); ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Plot 2 — actual vs predicted (test set)
ax = axes[0, 1]
ax.plot(dates_test, y_test,   "o-", color="#2E7D32", lw=1.8, label="Actual")
ax.plot(dates_test, lr_pred,  "s--", color="#E53935", lw=1.6,
        label=f"Linear (R²={lr_r2:.3f})")
ax.plot(dates_test, rr_pred,  "^:", color="#F57C00", lw=1.6,
        label=f"Ridge  (R²={rr_r2:.3f})")
ax.set_title("Test Period: Actual vs Predicted")
ax.set_ylabel("Sales"); ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Plot 3 — forecast
ax = axes[1, 0]
hist_tail = df.tail(12)
ax.plot(hist_tail["date"], hist_tail["sales"], color="#1976D2", lw=1.8, label="Historical")
ax.plot(future_dates, forecast, "o--", color="#9C27B0", lw=2, ms=7, label="Forecast 2025")
ax.axvline(df["date"].max(), color="gray", linestyle=":", alpha=0.7)
ax.fill_between(future_dates,
                forecast * 0.93, forecast * 1.07,
                color="#9C27B0", alpha=0.15, label="±7% Confidence")
ax.set_title("6-Month Ahead Forecast (Jan–Jun 2025)")
ax.set_ylabel("Sales"); ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Plot 4 — model metric comparison
ax = axes[1, 1]
models = ["Linear Reg.", "Ridge Reg."]
maes   = [lr_mae, rr_mae]
rmses  = [lr_rmse, rr_rmse]
r2s    = [lr_r2 * 100, rr_r2 * 100]
x = np.arange(len(models)); w = 0.25
ax.bar(x - w, maes,  w, label="MAE",       color="#1976D2", alpha=0.85)
ax.bar(x,     rmses, w, label="RMSE",      color="#E53935", alpha=0.85)
ax.bar(x + w, r2s,   w, label="R² × 100", color="#2E7D32", alpha=0.85)
ax.set_title("Model Performance Comparison")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Metric Value")
ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/predictive_analytics/results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Visualisation saved → results.png")

# ─────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
best = "Linear Regression" if lr_r2 >= rr_r2 else "Ridge Regression"
print(f"""
{'='*60}
SUMMARY
{'='*60}
Dataset       : 48 months historical sales (2021-2024)
Features      : Time index, Fourier seasonal terms (sin/cos)
Models        : Linear Regression, Ridge Regression

Results on held-out test set (Jul–Dec 2024):
  Linear Regression  →  MAE={lr_mae:.2f}, RMSE={lr_rmse:.2f}, R²={lr_r2:.4f}
  Ridge Regression   →  MAE={rr_mae:.2f}, RMSE={rr_rmse:.2f}, R²={rr_r2:.4f}

Best Model    : {best}
Forecast peak : {forecast_df.loc[forecast_df.Forecast.idxmax(), 'Date'].strftime('%b %Y')} \
({forecast_df.Forecast.max():.1f} units)

Techniques Used:
  ✓ Fourier seasonal decomposition (sin/cos transforms)
  ✓ Regularization via Ridge to prevent overfitting
  ✓ MAE, RMSE, R² evaluation metrics
  ✓ 6-month future forecast with confidence band
""")
