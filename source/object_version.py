import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Vasicek:
    def __init__(self, a: float = 0, b: float = 0, sigma: float = 0) -> None:
        self.a = a
        self.b = b
        self.sigma = sigma
        pass

    def loadData(self, path: str) -> None:
        self.data = pd.read_csv(filepath_or_buffer=path, index_col="date")
        pass

    def MLEfit(self) -> None:
        pass

    pass
