import os
import pandas as pd
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY / ALPACA_API_SECRET environment variables.")

SYMBOL = "AAPL"
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2026-01-15")

client = StockHistoricalDataClient(API_KEY, API_SECRET)
request_params = StockBarsRequest(
    symbol_or_symbols=[SYMBOL],
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
)

bars = client.get_stock_bars(request_params).df.reset_index()
bars = bars[bars["symbol"] == SYMBOL].copy()

bars.rename(
    columns={"timestamp": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"},
    inplace=True,
)
bars["Date"] = pd.to_datetime(bars["Date"])
bars.sort_values("Date", inplace=True)

# Features
bars["Return"] = bars["Close"].pct_change()
bars["Volatility_5"] = bars["Return"].rolling(5).std()
bars["MA_5"] = bars["Close"].rolling(5).mean()
bars["MA_10"] = bars["Close"].rolling(10).mean()
bars["MA_20"] = bars["Close"].rolling(20).mean()
bars["Intraday_Range"] = bars["High"] - bars["Low"]
bars["Price_Deviation"] = bars["Close"] - bars["MA_20"]
bars["Momentum_10"] = bars["Close"] - bars["Close"].shift(10)

bars["Target_Close_Next"] = bars["Close"].shift(-1)

df = bars.dropna().copy()

features = ["Return", "Volatility_5", "MA_5", "MA_10", "MA_20", "Intraday_Range", "Price_Deviation", "Momentum_10"]
X = df[features]
y = df["Target_Close_Next"]

# Time-aware split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df["Date"].iloc[split_idx:]

# Model
model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE (next-day close): {mae:.4f}")

# Optional: naive baseline (tomorrow = today)
naive_preds = df["Close"].iloc[split_idx:-1].values  # align sizes carefully
naive_mae = mean_absolute_error(y_test.values, naive_preds[: len(y_test.values)])
print(f"Naive MAE (tomorrow = today): {naive_mae:.4f}")

# Plot A: Close price
plt.figure(figsize=(11, 5))
plt.plot(df["Date"], df["Close"])
plt.title("AAPL Daily Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot B: Predicted vs actual
plt.figure(figsize=(11, 5))
plt.plot(dates_test, y_test.values, label="Actual Next-Day Close")
plt.plot(dates_test, preds, label="Predicted Next-Day Close")
plt.title("AAPL Next-Day Close: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Latest prediction
latest_X = X.iloc[[-1]]
next_close_pred = model.predict(latest_X)[0]
last_close = df["Close"].iloc[-1]
direction = "UP" if next_close_pred > last_close else "DOWN"
print(f"Last close: {last_close:.2f}")
print(f"Predicted next close: {next_close_pred:.2f} -> {direction}")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

features = ["Return", "Price_Deviation", "Momentum", "Intraday_Range"]
X = df[features].values
y = df["Target"].values  # next-day close

tscv = TimeSeriesSplit(n_splits=5)

maes, rmses = [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    maes.append(mean_absolute_error(y_test, preds))
    rmses.append(np.sqrt(mean_squared_error(y_test, preds)))

print("Walk-forward CV MAE (USD):", np.mean(maes), "+/-", np.std(maes))
print("Walk-forward CV RMSE (USD):", np.mean(rmses), "+/-", np.std(rmses))
