import numpy as np


class BlackScholesPDECn:
    """
    Crank–Nicolson solver for the Black–Scholes PDE:

        V_t = 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V

    on S in [0, S_max], t in [0, T], backward in time.
    """

    def __init__(self, S0, K, r, sigma, T,
                 S_max_factor=4.0, Ns=200, Nt=200):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

        self.S_max = S_max_factor * S0
        self.Ns = Ns
        self.Nt = Nt

        self.dS = self.S_max / Ns
        self.dt = T / Nt

        self.S = np.linspace(0.0, self.S_max, Ns + 1)

    def _setup_matrices(self):

        Ns = self.Ns
        dS = self.dS
        dt = self.dt
        r = self.r
        sigma = self.sigma

        A = np.zeros((Ns - 1, Ns - 1))
        B = np.zeros((Ns - 1, Ns - 1))

        for i in range(1, Ns):
            Si = i * dS


            alpha = 0.5 * sigma**2 * Si**2 / dS**2
            beta = 0.5 * r * Si / dS


            a = dt * (alpha - beta)
            b = dt * (-2.0 * alpha - r)
            c = dt * (alpha + beta)


            if i > 1:
                A[i - 1, i - 2] = -0.5 * a
            A[i - 1, i - 1] = 1.0 - 0.5 * b
            if i < Ns - 1:
                A[i - 1, i] = -0.5 * c


            if i > 1:
                B[i - 1, i - 2] = 0.5 * a
            B[i - 1, i - 1] = 1.0 + 0.5 * b
            if i < Ns - 1:
                B[i - 1, i] = 0.5 * c

        return A, B

    def solve_grid(self):
        Ns, Nt = self.Ns, self.Nt
        dt = self.dt
        r = self.r
        K = self.K

        S = self.S
        V = np.maximum(S - K, 0.0)

        A, B = self._setup_matrices()

        for n in range(Nt):
            t_next = self.T - (n + 1) * dt

            V_inner = V[1:Ns]

            rhs = B @ V_inner


            V_0 = 0.0
            V_Smax = self.S_max - K * np.exp(-r * t_next)

            rhs[0] -= A[0, 0] * 0.0
            rhs[-1] -= A[-1, -1] * 0.0

            rhs[-1] += 0.5 * dt * (0.5 * r * self.S_max / self.dS +
                                   0.5 *self.sigma**2 * self.S_max**2 / self.dS**2) * V_Smax

            V_new_inner = np.linalg.solve(A, rhs)

            V[0] = V_0
            V[1:Ns] = V_new_inner
            V[Ns] = V_Smax

        return V

    def price(self):
        V = self.solve_grid()
        return np.interp(self.S0, self.S, V)
