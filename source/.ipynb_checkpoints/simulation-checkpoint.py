import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: This file should include functions that take Vasicek parameters and return


def VasicekNormal(
    random_number: float = 0,
    current_r: float = 0,
    a: float = 0.3213874,
    b: float = 0.0172119,
    sigma: float = 0.0122468,
    delta: float = 1.0 / 250,
) -> float:
    """This function takes a random number (more efficient if simulated before, standard normal distribution)
    and transforms it into the next r_(t+1) given r_t (current_r). Since this is a normal distribution as well as shown in the MLE_Fit notebook
    or the report pdf, we can just transform a N(0,1) variable into N(mu, sigma^2) by scaling and translation. The formulas for this are
    according to the report as mentioned. This will be applied for entire arrays even though it only takes single values (made possible by numpy!)."""

    # Factor that is used often -> more efficient to compute only once
    nu = np.exp(-a * delta)

    # Calculate mean and variance of new normal distribution according to transition density derived in the report
    mean_new = current_r * nu + b * (1 - nu)
    var = (sigma**2) / (2 * a) * (1 - nu**2)

    # Calculate/sample new r value according to transition density (transform presampled N(0,1) number)
    r_new = np.sqrt(var) * random_number + mean_new

    # Return all possibly interesting data
    return r_new, mean_new, var


def SimulateProcess(
    a: float = 0.3213874,
    b: float = 0.0172119,
    sigma: float = 0.0122468,
    T: float = 1.0,
    n: int = 250,
) -> pd.DataFrame:
    """This function only generates one sample path of the Vasicek model but includes the mean and variance at each
    point in time for the next element normal distribution mainly used for intuition or further plots.
    T is the maturity (end of time horizon we consider) and n the amount of points between [0,T] that we want to sample.
    """
    # Define delta t -> important for transition density as variance and mean depend on it.
    delta = T / n

    # Declare numpy array of shape (n,3) (-> n rows and 3 columns)
    r = np.zeros(shape=(n, 3))  # r_s, means, variances

    # Initialize a generator that we can call for random numbers. Thats why VasicekNormal already takes a random number!
    #   If we would define this inside the function we would re-initialize in a for loop and throw it away after only generating one number:
    #     --> too slow, we call it ONCE and generate the values ahead of time. This only works exactly like this in our case because of the nice 
    #         behaviour of a neat transition density!
    ran = np.random.default_rng()

    # Call the random number generator for (pseudo)-random normal numbers with standard parameters. We call it ONCE and get n-1 many random values directly!
    #    (n-1 since for t=0 we have r_0 = 0 set so there are only n-1 many values left for which we need standard normal values)
    r_s = ran.normal(loc=0, scale=1, size=n - 1)

    # Generate n-1 many new elements:
    for i in range(1, n):
        # Take the previous r_t, the generated random number and parameters and compute the next elements:
        r_new, mean_new, var = VasicekNormal(
            random_number=r_s[i - 1],
            current_r=r[i - 1, 0],
            a=a,
            b=b,
            sigma=sigma,
            delta=delta,
        )
        # print(r_new, mean_new, var) (This was from debugging and getting a feel for it, not needed anymore)
        r[i, :] = r_new, mean_new, var

    # Convert to pandas.DataFrame:
    r = pd.DataFrame(data=r, columns=["r_t", "mean_t", "var"])

    return r


def SimulateTrajectories(
    a: float = 0.3213874,
    b: float = 0.0172119,
    sigma: float = 0.0122468,
    T: float = 1.0,
    n: int = 250,
    paths=10,
) -> pd.DataFrame:

    # Same as before:
    delta = T / n

    # Same as before but we dont want the mean and variances
    r = np.zeros(shape=(n, paths))  # (r_t)_{0<=t<=T} trajectory for each column, paths-many columns of iid trajectories

    # Same as before adjusted to how many paths/trajectories we want to simulate:
    ran = np.random.default_rng()
    r_s = ran.normal(loc=0, scale=1, size=(n - 1, paths))

    for i in range(1, n):
        # We only get r_new (save memory by not cluttering it with values we dont use using "..,_,_  = function()"
        r_new, _, _ = VasicekNormal(
            random_number=r_s[i - 1],
            current_r=r[i - 1],
            a=a,
            b=b,
            sigma=sigma,
            delta=delta,
        )
        # print(r[i, :], r_new)
        r[i, :] = r_new

    r = pd.DataFrame(r)
    return r


def PlotVasicek():

    r = SimulateTrajectories()

    plt.figure(figsize=(12, 5))
    plt.plot(r, label="Vasicek Trajectories")
    plt.xlabel("t")
    plt.ylabel("r_t")
    plt.title("Vasicek Trajectories")
    plt.grid(True)
    plt.savefig("vasicek_trajectories.png")

    return None


if __name__ == "__main__":
    PlotVasicek()
