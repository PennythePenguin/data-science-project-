import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AAPL Stock Prediction Dashboard", layout="centered")
st.title("AAPL Stock Prediction Dashboard")


# --------------------------
# 1) Cached data loading + feature engineering
# --------------------------
@st.cache_data
def load_data(csv_path: str = "AAPL_data.csv") -> pd.DataFrame:
    # Yahoo CSV has an extra "Ticker" row; skip it
    df = pd.read_csv(csv_path, skiprows=[1])

    # First column is actually Date (often called "Price" in Yahoo exports)
    df = df.rename(columns={df.columns[0]: "Date"})

    # Sometimes a stray row contains the string "Date"
    df = df[df["Date"] != "Date"]

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Feature engineering
    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Price_Deviation"] = df["Close"] - df["MA20"]
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["Intraday_Range"] = df["High"] - df["Low"]

    # Predict next-day close
    df["Target"] = df["Close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------
# 2) Cached model training
# --------------------------
@st.cache_resource
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# --------------------------
# 3) Cached Monte Carlo simulation
# --------------------------
@st.cache_data
def run_monte_carlo(
    last_close: float,
    mu: float,
    sigma: float,
    n_sims: int = 200,
    horizon: int = 30,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sims = np.zeros((n_sims, horizon), dtype=float)

    for i in range(n_sims):
        price = last_close
        for t in range(horizon):
            shock = rng.normal(mu, sigma)
            price = price * (1 + shock)
            sims[i, t] = price

    return sims


def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    nxt = date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:  # Sat/Sun
        nxt += pd.Timedelta(days=1)
    return nxt


# --------------------------
# Main app
# --------------------------
df = load_data("AAPL_data.csv")

features = ["Return", "Price_Deviation", "Momentum", "Intraday_Range"]
X = df[features]
y = df["Target"]

# Time-aware split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = train_model(X_train, y_train)

# Evaluate
pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
st.caption(f"Model test MAE (USD): {mae:.2f}")

# Latest prediction
prediction = model.predict(X.iloc[[-1]])[0]
last_close = df["Close"].iloc[-1]
delta = prediction - last_close

st.metric("Last Close (USD)", f"{last_close:.2f}")
st.metric("Predicted Next Close (USD)", f"{prediction:.2f}", delta=f"{delta:.2f} USD")
st.write("Direction:", "Going UP 📈" if delta > 0 else "Going DOWN 📉")

# --------------------------
# Plot 1: Price history + prediction point (next trading day)
# --------------------------
st.subheader("Stock Price")

last_date = df["Date"].iloc[-1]
pred_date = next_trading_day(last_date)

fig1 = plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.scatter([pred_date], [prediction], s=90, label="Predicted Next Close")
plt.plot([last_date, pred_date], [last_close, prediction], linestyle="--")
plt.annotate(
    f"{pred_date.date()}\n{prediction:.2f} USD",
    (pred_date, prediction),
    textcoords="offset points",
    xytext=(10, 10),
)

plt.title("AAPL Daily Closing Price + Next Trading Day Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

# --------------------------
# Plot 2: Monte Carlo (median + 10–90% band)
# --------------------------
st.subheader("Simulated Future Price Paths (Monte Carlo)")

mu = df["Return"].mean()
sigma = df["Return"].std()

# Simulate (cached)
simulations = run_monte_carlo(last_close, mu, sigma, n_sims=200, horizon=30, seed=42)

horizon = simulations.shape[1]
future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="B")[1:]

p10 = np.percentile(simulations, 10, axis=0)
p50 = np.percentile(simulations, 50, axis=0)
p90 = np.percentile(simulations, 90, axis=0)

fig2 = plt.figure(figsize=(10, 5))

# show a few sample paths (visual texture)
for path in simulations[:15]:
    plt.plot(future_dates, path, alpha=0.15)

plt.fill_between(future_dates, p10, p90, alpha=0.25, label="10–90% range")
plt.plot(future_dates, p50, linewidth=2, label="Median path")

plt.title("AAPL Monte Carlo Simulation (Next Trading Days)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# --------------------------
# Data preview
# --------------------------
st.subheader("Data Preview")
st.dataframe(df.tail(10))
