import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fit import MLEFit
from simulation import SimulateTrajectories


# This file contains all plot functions that we made. For some, we once again fit the model if we want do so
# by specifying the "fit" bool. If False, we have the possibility to compare the historical dataset values with simulations
# of trajectories that correspond to parameters that are different than those obtained through MLE estimation which could turn
# out to be useful.


def PlotData(data_path: str, save_path: str) -> None:
    # Load the cleaned data
    df = pd.read_csv(filepath_or_buffer=data_path, index_col="date")

    # Convert Index to datetime object:
    df.index = pd.to_datetime(df.index)
    # This is equivalent to setting parse_date=True in the pd.read_csv call which will be used from now on.

    # The data path is assumed to look something like this: ../data/RateName.csv.
    # To extract the Rate_Name, we do some cool string manipulation:
    # We do 3 splits:
    #   1.) split "/" and take the last one : we get RateName.csv
    #   2.) split"." and take the first one: we get RateName (no points in filenames allowed except for ending!)
    #   3.) split "_" and take the first one as some files are RateName_clean format: we get RateName in all cases.
    RateName = data_path.split("/")[-1].split(".")[0].split("_")[0]

    # Some if case for the title as SR corresponds to combined short rate data:
    if RateName == "SR":
        TitleName = "Combined DFF and SOFR"
    else:
        TitleName = RateName

    # Plot the short rate over time
    # Create matplotlib.pyplot object with specified width and height (in cm?)
    plt.figure(figsize=(12, 5))

    # Define what should be plotted on which axis, first argument on x and second on y per default
    plt.plot(df.index, df["rate_pct"], label=f"{TitleName} Short Rate")

    # Display a grid for better visability
    plt.grid()

    # Give x and y labels
    plt.xlabel("Date")
    plt.ylabel("Short Rate")

    # Title
    plt.title(f"{TitleName} Short Rate")

    # Save the figure object as png at specified path
    plt.savefig(fname=save_path)

    return None


################################


def DeterministicComparison(
    data_path: str,
    save_path: str,
    fit: bool = True,
    a_hat: float = 0.3213874,
    b_hat: float = 0.0172119,
    dt: float = 1 / 252,
) -> None:
    # This function is to show how the evolution of the deterministic(non-stochastick)
    # part of the model given a random r0 in the historical data.
    # The purpose is to show a tendency of r to drift towards the average parameter.

    # We use library matplotlib to make the plot of data, pandas for better data management and numpy for mathematical equations and operations
    df = pd.read_csv(filepath_or_buffer=data_path, index_col="date", parse_dates=True)

    if fit:
        # Fit model to data provided in path instead of provided (or omited) input parameters
        (a_hat, b_hat, _, _) = MLEFit(path=data_path)

    # Once again nice string splitting:
    RateName = data_path.split("/")[-1].split(".")[0].split("_")[0]

    r_hist = df["rate_pct"]

    # Time step in years (daily data)
    T = len(r_hist)
    time_grid = np.arange(T) * dt

    # Deterministic expected trajectory function, see report for where this comes from
    #   (we only need this function inside this scope, could be defined outside as well)
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
    plt.plot(
        plot_idx, plot_hist, color="black", alpha=0.6, label="Historical Short Rate"
    )
    plt.plot(
        r_hist.index,
        traj1,
        color="blue",
        label=f"Expected Trajectory from r0={r0_1:.3f}",
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
    plt.title(f"Historical vs Expected Vasicek {RateName}")
    plt.grid(True)
    plt.legend()

    # plt.show()
    plt.savefig(fname=save_path)
    return None


################################


def HistandSim(
    data_path: str,
    save_path: str,
    fit: bool = True,
    a_hat: float = 0.3213874,
    b_hat: float = 0.0172119,
    sigma_hat: float = 0.0122468,
    dt: float = 1 / 252,
) -> None:

    plt.style.use("default")

    df = pd.read_csv(filepath_or_buffer=data_path, index_col="date", parse_dates=True)

    r_hist = df["rate_pct"]

    if fit:
        # Fit model to data provided in path instead of provided (or omited) input parameters
        (a_hat, b_hat, sigma_hat, _) = MLEFit(path=data_path)

    # Simulation parameters
    T_days = len(r_hist)
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
    plt.savefig(fname=save_path)
    return None


def PlotVasicekRandom(
    data_path: str,
    save_path: str,
    fit: bool = True,
    a_hat: float = 0.3213874,
    b_hat: float = 0.0172119,
    sigma_hat: float = 0.0122468,
) -> None:

    # Once again, as above if wanted you can fit again
    if fit:
        (a_hat, b_hat, sigma_hat, _) = MLEFit(path=data_path)

    r = SimulateTrajectories(a=a_hat, b=b_hat, sigma=sigma_hat)

    plt.figure(figsize=(12, 5))
    plt.plot(r, label="Vasicek Trajectories")
    plt.xlabel("t")
    plt.ylabel("r_t")
    plt.title("Vasicek Trajectories")
    plt.grid(True)
    plt.savefig(fname=save_path)

    return None


if __name__ == "__main__":
    # Predefine paths (we are too lazy to type this every time...)
    datasets = ["SOFR_clean", "DFF_clean", "SR"]

    DataPaths = [f"../data/processed/{name}.csv" for name in datasets]
    IMGPaths = [f"../data/plots/{name}/" for name in datasets]

    # Call functions to generate all the plots in data/plots directory:
    # (For most functions here we fit the model multiple times. This is not as efficient as it could be but thats only because
    # we call all functions at once. If we would only call one or two, this behaviour is convenient and the MLEFit function does not
    # take long)
    for datapath, imgpath in zip(DataPaths, IMGPaths):
        print(f"\n Creating All Plots for {datapath} dataset saved at {imgpath} \n")
        PlotData(data_path=datapath, save_path=imgpath + "Data.png")
        DeterministicComparison(
            data_path=datapath, save_path=imgpath + "Deterministic.png"
        )
        HistandSim(data_path=datapath, save_path=imgpath + "Historic_and_Sim.png")
        PlotVasicekRandom(data_path=datapath, save_path=imgpath + "Random_Sim.png")
