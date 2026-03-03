import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AAPL_with_indicators.csv")

# Rename the date column (your file uses "Price" but it contains dates)
df = df.rename(columns={"Price": "Date"})

# Robust date parsing: invalid strings (like "Ticker") -> NaT, then dropped
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# Sort time series
df = df.sort_values("Date").reset_index(drop=True)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"], label="AAPL Close Price")

plt.title("AAPL Daily Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save (guaranteed to work even if pop-up doesn't)
plt.savefig("aapl_price.png", dpi=200)
print("Saved plot to aapl_price.png")