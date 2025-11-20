import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: This file should include functions that take Vasicek parameters and return
# TODO: plots that show sample paths, resulting distribution function (maybe estimate Euler Maruyama error)


def VasicekNormal(
    random_number: float = 0,
    current_r: float = 0,
    a: float = 0.3039,
    b: float = 0.0312341,
    sigma: float = 0.00911741,
    delta: float = 1.0 / 250,
):
    nu = np.exp(-a * delta)
    mean_new = current_r * nu + b * (1 - nu)
    var = (sigma**2) / (2 * a) * (1 - nu**2)

    r_new = np.sqrt(var) * random_number + mean_new

    return r_new, mean_new, var


def SimulateProcess(
    a: float = 0.3039,
    b: float = 0.0312341,
    sigma: float = 0.00911741,
    T: float = 1.0,
    n: int = 250,
):
    delta = T / n

    r = np.zeros(shape=(n, 3))  # r_s, means, variances

    ran = np.random.default_rng()
    r_s = ran.normal(loc=0, scale=1, size=n - 1)
    for i in range(1, n):
        # print(i - 1)
        # print(r_s[i - 1])
        r_new, mean_new, var = VasicekNormal(
            random_number=r_s[i - 1],
            current_r=r[i - 1, 0],
            a=a,
            b=b,
            sigma=sigma,
            delta=delta,
        )
        # print(r_new, mean_new, var)
        r[i, :] = r_new, mean_new, var

    return r


def PlotVasicek():

    r = SimulateProcess()

    return None


if __name__ == "__main__":
    r = SimulateProcess()
    print(r)
