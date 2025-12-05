import pandas as pd
import numpy as np
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



################################



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


################################



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
