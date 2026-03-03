import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

NO_CHANGE_THRESHOLD = 0.0005  # 0.05%


def direction_label(value, threshold=NO_CHANGE_THRESHOLD):
    if value > threshold:
        return "increasing"
    if value < -threshold:
        return "decreasing"
    return "no change"


# =====================================================
# Load data
# =====================================================
df = pd.read_csv("AAPL_with_indicators.csv")

df = df.rename(columns={"Price": "Date"})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
df_all = df.copy()

# =====================================================
# Target: next-day return
# =====================================================
df["Return_Next"] = df["Return"].shift(-1)
df = df.dropna(subset=["Return_Next"])

print("Data ready. Rows:", len(df))

# =====================================================
# Features
# =====================================================
features = ["Momentum", "Return", "Price_Deviation"]

X = df[features]
y = df["Return_Next"]

# Time-aware split 
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test = df["Date"].iloc[split:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# =====================================================
# Model pipeline (scaling + linear regression)
# =====================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

model.fit(X_train, y_train)

# =====================================================
# Predictions
# =====================================================
preds = model.predict(X_test)

# =====================================================
# Metrics
# =====================================================
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# Direction output (increasing / decreasing / no change)
#gives me the outcome
pred_dirs = [direction_label(v) for v in preds]
actual_dirs = [direction_label(v) for v in y_test.values]
direction_acc = np.mean(np.array(pred_dirs) == np.array(actual_dirs))

print(f"Direction threshold: +/-{NO_CHANGE_THRESHOLD:.4%}")
print(f"Direction accuracy: {direction_acc:.2%}")

direction_out = pd.DataFrame({
    "Date": dates_test.values,
    "Predicted_Return_Next": preds,
    "Predicted_Direction": pred_dirs,
    "Actual_Direction": actual_dirs
})

print("\nRecent direction predictions:")
print(direction_out.tail(10).to_string(index=False))

# Final next-day direction forecast from the latest available day
model.fit(X, y)
latest_features = df_all[features].iloc[[-1]]
latest_date = df_all["Date"].iloc[-1]
latest_pred_return = float(model.predict(latest_features)[0])
latest_pred_direction = direction_label(latest_pred_return)

print("\nLatest forecast:")
print(f"Based on {latest_date.date()}, next-day return is predicted as: {latest_pred_direction}")
print(f"Predicted return value: {latest_pred_return:.6f}")

# =====================================================
# Plot — Actual vs Predicted
# =====================================================
plt.figure(figsize=(10,5))
plt.plot(dates_test, y_test.values, label="Actual")
plt.plot(dates_test, preds, label="Predicted")
plt.title("Linear Regression: Actual vs Predicted Return")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("linear_actual_vs_pred.png", dpi=200)
plt.close()

# =====================================================
# Residual plot
# =====================================================
residuals = y_test.values - preds

plt.figure(figsize=(6,5))
plt.scatter(preds, residuals, alpha=0.4)
plt.axhline(0)
plt.title("Residual Plot — Linear Regression")
plt.xlabel("Predicted Return")
plt.ylabel("Residual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("linear_residual_plot.png", dpi=200)
plt.close()

print("✅ ")
