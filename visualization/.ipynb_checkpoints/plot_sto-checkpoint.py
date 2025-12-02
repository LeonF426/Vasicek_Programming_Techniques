import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This file purpose is to show how the evolution of the model given a random r0 in the historical data

plt.style.use("default")

df = pd.read_csv("../data/DFF_clean.csv", index_col="date", parse_dates=True)

r_hist = df["rate_pct"]

# Vasicek parameters (MLE estimates)
# a_hat = 0.3039
# b_hat = 0.0312341
# sigma_hat = 0.00911741

a_hat = 0.3213874
b_hat = 0.0172119
sigma_hat = 0.0122468

# Simulation parameters
T_days = len(r_hist)
dt = 1 / 252
n_paths = 3  # number of stochastic paths
downsample_step = 10  # plot only every 5th point


# Function to simulate a Vasicek path
def vasicek_sim(r0, a, b, sigma, dt, n):
    r = np.zeros(n)
    r[0] = r0
    for i in range(1, n):
        dr = a * (b - r[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        r[i] = r[i - 1] + dr
    return r


time_axis = np.arange(T_days) * dt

r_avg = b_hat

plot_hist = r_hist[::downsample_step]
plot_idx = r_hist.index[::downsample_step]

# Plot
plt.figure(figsize=(12, 6))

# Plot original historical short rate in black
plt.plot(plot_idx, plot_hist, color="black", alpha=0.8, label="Path historical")

# Simulate and plot stochastic paths
colors = plt.cm.tab10(np.linspace(0, 1, n_paths + 3))
r0 = r_hist.iloc[0]

for i in range(n_paths):
    path = vasicek_sim(r0, a_hat, b_hat, sigma_hat, dt, T_days)
    plt.plot(
        r_hist.index[::downsample_step],
        path[::downsample_step],
        color=colors[i + 3],
        alpha=0.8,
        lw=0.8,
        label=f"Path {i+1}",
    )
plt.axhline(
    r_avg, color="grey", linestyle="--", label=f"Historical Average {r_avg:.4f}"
)

plt.xlabel("Date")
plt.ylabel("Short Rate")
plt.title("Vasicek Model: Stochastic Short Rate Paths vs Historical")
plt.grid(True)
plt.legend(loc="lower right")
# plt.show()
plt.savefig("sto.png")
