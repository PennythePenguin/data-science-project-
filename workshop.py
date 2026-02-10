import pandas as pd
df = pd.read_csv("AAPL_data.csv")

numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
close_summary = df['Close'].agg(['mean','std','min','max'])
print("Close price summary:")
print(close_summary)

volume_summary = df['Volume'].agg(['mean','std','min','max'])
print("Volume summary:")
print(volume_summary)

df['High_low'] = df['High'] - df['Low']
high_low_summary = df['High_low'].agg(['mean','std','min','max'])
print("High-low range summary:")
print(high_low_summary)