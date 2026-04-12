import numpy as np
from math import log, sqrt, exp
from math import erf

from .blackscholespde import BlackScholesPDECn


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


class BlackScholesEngine:
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T


    def _d1_d2(self):
        S0, K, r, sigma, T = self.S0, self.K, self.r, self.sigma, self.T
        d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return d1, d2

    def bs_call(self) -> float:
        d1, d2 = self._d1_d2()
        S0, K, r, T = self.S0, self.K, self.r, self.T
        return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

    def bs_delta(self) -> float:
        d1, _ = self._d1_d2()
        return norm_cdf(d1)



    def pde_call(self, S_max_factor: float = 4.0, Nx: int = 200, Nt: int = 200) -> float:
        solver = BlackScholesPDECn(
            S0=self.S0,
            K=self.K,
            r=self.r,
            sigma=self.sigma,
            T=self.T,
            S_max_factor=S_max_factor,
            Ns=Nx,
            Nt=Nt,
        )
        return solver.price()



    def mc_call(self, n_paths: int = 50_000, n_steps: int = 100, seed: int | None = None):
        S0, K, r, sigma, T = self.S0, self.K, self.r, self.sigma, self.T

        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        nudt = (r - 0.5 * sigma**2) * dt
        sigsdt = sigma * sqrt(dt)

        S = np.full(n_paths, S0, dtype=float)

        for _ in range(n_steps):
            z = np.random.randn(n_paths)
            S *= np.exp(nudt + sigsdt * z)

        payoffs = np.maximum(S - K, 0.0)
        disc_payoffs = np.exp(-r * T) * payoffs

        price = disc_payoffs.mean()
        std_err = disc_payoffs.std(ddof=1) / np.sqrt(n_paths)

        return price, std_err
