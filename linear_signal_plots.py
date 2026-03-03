import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

df = pd.read_csv("AAPL_with_indicators.csv")

df = df.rename(columns={"Price": "Date"})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Create next-day return (TARGET)
df["Return_Next"] = df["Return"].shift(-1)
df = df.dropna(subset=["Return_Next"])

print("Data ready. Rows:", len(df))

# Helper function to plot scatter + best-fit line
def scatter_with_fit(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, alpha=0.4)

    # ⭐ Best-fit line
    m, b = np.polyfit(x, y, 1)

    # sort for smooth line
    x_sorted = np.sort(x)
    y_fit = m * x_sorted + b
    plt.plot(x_sorted, y_fit)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# PLOT 1 — Momentum vs Return_Next
scatter_with_fit(
    df["Momentum"].values,
    df["Return_Next"].values,
    "Momentum vs Next-Day Return",
    "Momentum (today)",
    "Return (tomorrow)",
    "scatter_momentum_vs_return_next.png"
)


# PLOT 2 — Price Deviation vs Return_Next
scatter_with_fit(
    df["Price_Deviation"].values,
    df["Return_Next"].values,
    "Price Deviation vs Next-Day Return",
    "Price Deviation (today)",
    "Return (tomorrow)",
    "scatter_price_dev_vs_return_next.png"
)


# PLOT 3 — Return vs Return_Next
scatter_with_fit(
    df["Return"].values,
    df["Return_Next"].values,
    "Return vs Next-Day Return ",
    "Return (today)",
    "Return (tomorrow)",
    "scatter_return_vs_return_next.png"
)

print("done")