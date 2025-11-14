import numpy as np
import scipy
import pandas as pd

data = pd.read_csv("../data/dgs10_clean.csv", index_col="date")
# print(data)

# define day-delta on the basis of which we will simulate the SDE dynamics (discretization step):
dt = 1 / 252

"""
As described in the original paper from Vasicek (AN EQUILIBRIUM CHARACTERIZATION OF THE TERM STRUCTURE - 1977) EQ (24):
  dr = alpha(gamma-r)dt + pdW_t
  with alpha > 0. This is a specific OU (Ornstein Uhlenbeck) process which has an exact (and strong) solution.
"""

"""
We will fit these parameters using MLE (Maximum Likelihood Estimator), thus searching for parameters
that explain the observed data with the highest probability. We therefore define the closed form for the
negative likelihood for our specific case and minimize using the scipy library. 
We will use scipy.optimize.minimize:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""

r = data.values  # array of observed short-rate proxy


def NegLLVasicek(params: tuple, r, dt: float) -> float:
    # params are in unconstrained space? We'll impose constraints in optimizer
    a, b, sigma = params
    # if a <= 0 or sigma <= 0:
    #     return 1e10
    phi = np.exp(-a * dt)
    var = (sigma**2 / (2 * a)) * (1 - phi**2)
    # compute conditional means
    mu = phi * r[:-1] + (1 - phi) * b
    # gaussian log-likelihood
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + ((r[1:] - mu) ** 2) / var)
    return -ll  # negative log-likelihood


def PreFit():
    return None


def MLEFit():
    return None


# initial guess from OLS AR(1) quick method
# r_{t+1} = alpha + phi*r_t + eps
# X = np.column_stack([np.ones(len(r) - 1), r[:-1]])
#
# Y = r[1:]
# beta = np.linalg.lstsq(X, Y, rcond=None)[0]
# alpha_hat, phi_hat = beta[0], beta[1]
# a_init = -np.log(np.clip(phi_hat, 1e-8, 0.9999)) / dt
# b_init = alpha_hat / (1 - phi_hat)
# resid = Y - X.dot(beta)
# var_eps = resid.var(ddof=2)
# sigma_init = np.sqrt(2 * a_init * var_eps / (1 - np.exp(-2 * a_init * dt)))
#
# start = np.array([a_init[0], b_init[0], sigma_init[0]])
start = np.array([1, 1, 1])
print(start)
# bounds: a>1e-6, sigma>1e-8, b unconstrained (but sensible)
bnds = [(1e-6, 5.0), (None, None), (1e-8, 2.0)]

bnds = [(1e-9, 10.0), (None, None), (1e-9, 10.0)]

opt = scipy.optimize.minimize(
    NegLLVasicek, start, args=(r, dt), method="L-BFGS-B", bounds=bnds
)
if not opt.success:
    raise RuntimeError("MLE optimization failed: " + opt.message)

a_mle, b_mle, sigma_mle = opt.x
print(
    "MLE estimates: a = %.6g, b = %.6g, sigma = %.6g, fun-value = %.6g"
    % (a_mle, b_mle, sigma_mle, opt.fun)
)
