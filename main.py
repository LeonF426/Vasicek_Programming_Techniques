"""This file is the main.py file which is inteded to calculate one entire 'run'
of our project:
    - Explain the model and our goal --> pdf report
    - Read in Data and do cleanup --> functionalities in ./data/data_cleaning.py
    - Read in cleaned Data, use model structure and fit parameters using MLE --> ./source/fit.py
    - Use parameters and compute plots of distribution and sample trajectories --> ./source/simulation.py
    - Use trajectories to price derivatives --> ./source/pricing.py
    - Use derived model from main reference article to approximate term structure"""

import pandas as pd
from source import fit, data_cleaning, simulation


def main():
    print("Hello from pt-project!")

    # Run the entire project pipeline:

    # 1) Data prep (for SOFR and DFF):
    data_cleaning.DataCleanup(filename="./data/DFF.csv")  # --> ./data/DFF_clean.csv
    data_cleaning.DataCleanup(filename="./data/SOFR.csv")  # --> ./data/SOFR_clean.csv
    data_cleaning.JoinData()  # --> ./data/.SR.csv combined Short Rate Data (discussed in dataprep notebook)

    # 2) Fit Vasicek Parameters using MLE:
    dff_mle = fit.MLEFit(path="./data/DFF_clean.csv")
    sofr_mle = fit.MLEFit(path="./data/SOFR_clean.csv")
    shortrate_mle = fit.MLEFit(path="./data/SR.csv")

    # 3) Simulate Vasicek trajectories for maturity T and n many discretization points:
    T = 1.0  # 1year
    n = 252  # 252 days per year
    paths = 10  # amount of trajectories simulated

    dff_paths = simulation.SimulateTrajectories(
        a=dff_mle[0], b=dff_mle[1], sigma=dff_mle[2], T=T, n=n, paths=paths
    )
    sofr_paths = simulation.SimulateTrajectories(
        a=sofr_mle[0], b=sofr_mle[1], sigma=sofr_mle[2], T=T, n=n, paths=paths
    )
    shortrate_paths = simulation.SimulateTrajectories(
        a=shortrate_mle[0],
        b=shortrate_mle[1],
        sigma=shortrate_mle[2],
        T=T,
        n=n,
        paths=paths,
    )


if __name__ == "__main__":
    main()
