import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This file purpose is to show how the evolution of the deterministic(non-stochastick) part of the model given a random r0 in the historical data. The purpose is to show a tendency of r to drift towards the average parameter.

# We use library matplotlib to make the plot of data, pandas for better data management and numpy for mathematical equations and operations
df = pd.read_csv("../data/DFF_clean.csv", index_col="date", parse_dates=True)

r_hist = df["rate_pct"]

# Vasicek parameters (MLE estimates)
# a_hat = 0.3039
# b_hat = 0.0312341
# sigma_hat = 0.00911741

a_hat = 0.3213874
b_hat = 0.0172119
sigma_hat = 0.0122468

# Time step in years (daily data)
dt = 1 / 252
T = len(r_hist)
time_grid = np.arange(T) * dt


# Deterministic expected trajectory function
def expected_trajectory(r0, a, b, t):
    return r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))


# Trajectory starting from the first observation
r0_1 = r_hist.iloc[0]
traj1 = expected_trajectory(r0_1, a_hat, b_hat, time_grid)

# Trajectory starting from a random observation
start_idx = np.random.randint(0, len(r_hist))
r0_2 = r_hist.iloc[start_idx]
T2 = len(r_hist) - start_idx
time_grid2 = np.arange(T2) * dt
traj2 = expected_trajectory(r0_2, a_hat, b_hat, time_grid2)

# Optimized model historical average
r_avg = b_hat

# Downsample for plotting historical data
plot_hist = r_hist[::5]
plot_idx = r_hist.index[::5]

# Plot
plt.figure(figsize=(12, 4))
plt.plot(plot_idx, plot_hist, color="black", alpha=0.6, label="Historical Short Rate")
plt.plot(
    r_hist.index, traj1, color="blue", label=f"Expected Trajectory from r0={r0_1:.3f}"
)
plt.plot(
    r_hist.index[start_idx:],
    traj2,
    color="purple",
    label=f"Expected Trajectory from r0={r0_2:.3f}",
)
plt.axhline(
    r_avg, color="grey", linestyle="--", label=f"Historical Average {r_avg:.4f}"
)
plt.xlabel("Date")
plt.ylabel("Short Rate")
plt.title("Historical vs Expected Vasicek  DGS10 Yield Trajectories")
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig("vasi_try1.png")
