import pandas as pd
import matplotlib.pyplot as plt


# This file is to get the plot for the cleaned data
# We use library matplotlib to make the plot of data, pandas for better data management

# 1. Load the cleaned data
df = pd.read_csv("../data/DFF_clean.csv", index_col="date")

# Check the first few rows
print(df.head())


# 2. Plot the short rate over time
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["rate_pct"], label="10-Year Treasury Rate")
plt.xlabel("Date")
plt.ylabel("Short Rate (decimal)")
plt.title("Historical 10-Year Treasury Rate")
plt.savefig("DFF_plot.png")
