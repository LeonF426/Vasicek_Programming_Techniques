# Fast package to handle array wide computations:
import numpy as np

# Import enhanced type checking functionalities for arrays to use as input datatype for observations:
from numpy.typing import NDArray

# Package that we will use for minimization:
import scipy

# Package for data structures and reading/saving data:
import pandas as pd


def NegLLVasicek(params: tuple, r: NDArray[np.float64], dt: float) -> float:
    """Function that evaluates the negative log-likelihood function specifically in our underlying Vasicek model."""
    # Assign just as we described above
    a, b, sigma = params

    # This term is observed multiple times, compute once for efficiency
    phi = np.exp(-a * dt)

    # Compute variance (independant for r!)
    var = (sigma**2 / (2 * a)) * (1 - phi**2)

    # compute conditional means (for the entire array! mu is an array with the conditional means!)
    mu = phi * r[:-1] + (1 - phi) * b

    # gaussian log-likelihood (just as term above but taking the sum directly)
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + ((r[1:] - mu) ** 2) / var)
    return -ll  # negative log-likelihood


"""
In contrast to the jupyter notebook, we will wrap the remaining functionalities into a function such that
it can be imported into other python files. 
"""

# TODO: include procedure to find good starting values!


def MLEFit(
    path: str = "../data/SR.csv",
    dt: float = 1 / 252,
    start=np.array([0.5, None, 0.5]),
    bounds: list = [
        (1e-9, 1),  # bounds for a --> a > 0
        (-1, 1),  # bounds for b --> unconstrained
        (1e-9, 1),  # bounds for sigma --> sigma > 0
    ],
) -> tuple[float, float, float, float]:
    """
    This function has the exact same funcitonalities as the code shown in the MLE_Fit.ipynb notebook.
    We just wrap it into a function such that it can be imported in other python files. We set the values we use
    as standard inputs but they can be changed if needed."""

    # Use pandas functionality to read in the cleaned dgs10 data:
    data = pd.read_csv(filepath_or_buffer=path, index_col="date")

    # Print first 10 rows to take a look:
    # print(data.head(10))

    # define day-delta on the basis of which we will simulate the SDE dynamics (discretization step assumed to be average business day per year):
    # dt = 1 / 252 HERE NOW AS INPUT!

    # Extract the values of the 'data' Dataframe into an array (DataFrame method returns np.array)
    r = data.values
    # print(type(r))

    # # Some starting parameters
    # start = np.array([1, 1, 1]) HERE NOW AS INPUT!
    start[1] = r.mean()

    # Justification for these bounds are given by the parameter description in the introduction to this model
    # bnds = [
    #     (1e-9, 10.0),  # bounds for a --> a > 0
    #     (None, None),  # bounds for b --> unconstrained
    #     (1e-9, 10.0),  # bounds for sigma --> sigma > 0
    # ]  NOW INPUT!

    # This will return an 'OptimizeResult' object that has multiple attributes that we call later
    opt = scipy.optimize.minimize(
        fun=NegLLVasicek, x0=start, args=(r, dt), method="L-BFGS-B", bounds=bounds
    )

    # opt.success is an attribute that signals if the procedure has run without errors
    if not opt.success:
        raise RuntimeError("MLE optimization failed: " + opt.message)

    # Multi-assignment from opt.x attribute containing optimization result array + opt.fun gives log-likelihood function value for optimized values
    a_mle, b_mle, sigma_mle = opt.x
    print(
        "MLE estimates: a = %.6g, b = %.6g, sigma = %.6g, fun-value = %.6g"
        % (a_mle, b_mle, sigma_mle, opt.fun)
    )

    return (a_mle, b_mle, sigma_mle, opt.fun)


if __name__ == "__main__":
    MLEFit(path="../data/SOFR_clean.csv")
    MLEFit(path="../data/DFF_clean.csv")
    MLEFit(path="../data/SR.csv")
