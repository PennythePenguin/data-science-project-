import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n========== STEP 1: LOAD DATA ==========")

df = pd.read_csv("AAPL_with_indicators.csv")

# Fix date
df = df.rename(columns={"Price": "Date"})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

print("Rows:", len(df))
print("Columns:", list(df.columns))


# =====================================================
# STEP 2 — Required columns check
# =====================================================
print("\n========== STEP 2: COLUMN CHECK ==========")

required_cols = ["Momentum", "Return", "Price_Deviation", "Close"]

for col in required_cols:
    if col in df.columns:
        print(f"✅ {col} exists")
    else:
        print(f"❌ MISSING COLUMN: {col}")


# =====================================================
# STEP 3 — Missing values
# =====================================================
print("\n========== STEP 3: MISSING VALUES ==========")

na_counts = df.isna().sum()
print(na_counts[na_counts > 0])


# =====================================================
# STEP 4 — Create target safely
# =====================================================
print("\n========== STEP 4: TARGET CREATION ==========")

df["Return_Next"] = df["Return"].shift(-1)
df = df.dropna(subset=["Return_Next"])

print("After shift rows:", len(df))


# =====================================================
# STEP 5 — Basic statistics sanity
# =====================================================
print("\n========== STEP 5: BASIC STATS ==========")

print("Return mean:", df["Return"].mean())
print("Return std:", df["Return"].std())
print("Return_Next mean:", df["Return_Next"].mean())
print("Return_Next std:", df["Return_Next"].std())


# =====================================================
# STEP 6 — Train/test split check
# =====================================================
print("\n========== STEP 6: TIME SPLIT CHECK ==========")

features = ["Momentum", "Return", "Price_Deviation"]

X = df[features]
y = df["Return_Next"]

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Train last date:", df["Date"].iloc[split-1])
print("Test first date:", df["Date"].iloc[split])

if df["Date"].iloc[split] > df["Date"].iloc[split-1]:
    print("✅ Time order correct (no leakage)")
else:
    print("❌ WARNING: time order issue")


# =====================================================
# STEP 7 — Naive baseline check
# =====================================================
print("\n========== STEP 7: NAIVE BASELINE ==========")

# naive prediction: tomorrow return = today return
naive_preds = df["Return"].iloc[split:-1].values
naive_true = y_test.values[:len(naive_preds)]

naive_mae = mean_absolute_error(naive_true, naive_preds)
naive_rmse = np.sqrt(mean_squared_error(naive_true, naive_preds))

print(f"Naive MAE: {naive_mae:.6f}")
print(f"Naive RMSE: {naive_rmse:.6f}")

print("\n========== CHECK COMPLETE ==========")