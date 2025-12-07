"""This file is the main.py file which is inteded to calculate one entire 'run'
of our project:
    - Explain the model,theory and present implementations --> ./notebooks/report.ipynb
    - Read in Data and do cleanup --> functionalities in ./source/data_cleaning.py
    - Read in cleaned Data, use model structure and fit parameters using MLE --> ./source/fit.py
    - Use parameters and compute plots of distribution and sample trajectories --> ./source/simulation.py"""

import pandas as pd
from source import fit, data_cleaning, simulation


def main():
    print("Hello from pt-project!")

    # Run the entire project pipeline:

    # 1) Data prep (for SOFR and DFF):
    data_cleaning.DataCleanup(filename="./data/raw/DFF.csv")  # --> ./data/processed/DFF_clean.csv
    data_cleaning.DataCleanup(filename="./data/raw/SOFR.csv")  # --> ./data/processed/SOFR_clean.csv
    data_cleaning.JoinData()  # --> ./data/processed/SR.csv combined Short Rate Data (discussed in dataprep notebook)

    # 2) Fit Vasicek Parameters using MLE:
    a_DFF, b_DFF, sigma_DFF,_ = fit.MLEFit(path="./data/processed/DFF_clean.csv")
    a_SOFR, b_SOFR, sigma_SOFR,_ = fit.MLEFit(path="./data/processed/SOFR_clean.csv")
    a_SR, b_SR, sigma_SR,_ = fit.MLEFit(path="./data/processed/SR.csv")

    # 3) Simulate Vasicek trajectories for maturity T and n many discretization points:
    T = 1.0  # 1year
    n = 252  # 252 days per year
    paths = 10  # amount of trajectories simulated

    dff_paths = simulation.SimulateTrajectories(
        a=a_DFF, b=b_DFF, sigma=sigma_DFF, T=T, n=n, paths=paths
    )
    sofr_paths = simulation.SimulateTrajectories(
        a=a_SOFR, b=b_SOFR, sigma=sigma_SOFR, T=T, n=n, paths=paths
    )
    shortrate_paths = simulation.SimulateTrajectories(
        a=a_SR,
        b=b_SR,
        sigma=sigma_SR,
        T=T,
        n=n,
        paths=paths,
    )


if __name__ == "__main__":
    main()
