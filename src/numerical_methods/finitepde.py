import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math


class ExplicitHeat1D:
    def __init__(self, L=1.0, Nx=100, T=0.1, Nt=100, alpha=1.0):
        self.L = L
        self.Nx = Nx
        self.T = T
        self.Nt = Nt
        self.alpha = alpha

        self.dx = L / Nx
        self.dt = T / Nt
        self.lambda_cfl = alpha * self.dt / self.dx ** 2

        self.x = [j * self.dx for j in range(Nx + 1)]
        self.t = [n * self.dt for n in range(Nt + 1)]

        self.U = None

    def initial_condition(self, x):
        # Gaussian bump
        return math.exp(-100 * (x - 0.5 * self.L) ** 2)

    def apply_boundary(self, u, t):
        u[0] = 0.0
        u[-1] = 0.0
        return u

    def solve(self):
        Nx, Nt = self.Nx, self.Nt
        lam = self.lambda_cfl

        U = [[0.0] * (Nx + 1) for _ in range(Nt + 1)]

        for j in range(Nx + 1):
            U[0][j] = self.initial_condition(self.x[j])
        U[0] = self.apply_boundary(U[0], self.t[0])

        for n in range(Nt):
            u = U[n]
            u_new = u.copy()

            for j in range(1, Nx):
                u_new[j] = u[j] + lam * (u[j + 1] - 2 * u[j] + u[j - 1])

            u_new = self.apply_boundary(u_new, self.t[n + 1])
            U[n + 1] = u_new

        self.U = U
        return U

    def plot(self, times=[0.0, None]):
        if self.U is None:
            raise RuntimeError("Call solve() before plot().")

        plt.figure(figsize=(7, 4))

        for tau in times:
            if tau is None:
                idx = -1
            else:
                idx = min(range(len(self.t)), key=lambda i: abs(self.t[i] - tau))

            plt.plot(self.x, self.U[idx], label=f"t={self.t[idx]:.4f}")

        plt.title(f"Explicit Heat — λ = {self.lambda_cfl:.3f}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()


class ImplicitHeat1D:
    def __init__(self, L=1.0, Nx=100, T=0.1, Nt=100, alpha=1.0):
        self.L = L
        self.Nx = Nx
        self.T = T
        self.Nt = Nt
        self.alpha = alpha

        self.dx = L / Nx
        self.dt = T / Nt
        self.lambda_cfl = alpha * self.dt / self.dx ** 2

        self.x = [j * self.dx for j in range(Nx + 1)]
        self.t = [n * self.dt for n in range(Nt + 1)]

        self.U = None

    def initial_condition(self, x):
        return math.exp(-100 * (x - 0.5 * self.L) ** 2)

    def apply_boundary(self, u, t):
        u[0] = 0.0
        u[-1] = 0.0
        return u

    def solve_tridiagonal(self, a, b, c, d):
        n = len(d)
        c_star = [0.0] * n
        d_star = [0.0] * n

        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_star[i - 1]
            c_star[i] = c[i] / denom if i < n - 1 else 0.0
            d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

        x = [0.0] * n
        x[-1] = d_star[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def solve(self):
        Nx, Nt = self.Nx, self.Nt
        lam = self.lambda_cfl

        U = [[0.0] * (Nx + 1) for _ in range(Nt + 1)]

        for j in range(Nx + 1):
            U[0][j] = self.initial_condition(self.x[j])
        U[0] = self.apply_boundary(U[0], self.t[0])

        a = [-lam] * (Nx - 1)
        b = [1 + 2 * lam] * (Nx - 1)
        c = [-lam] * (Nx - 1)

        for n in range(Nt):
            rhs = U[n][1:-1]
            u_new_inner = self.solve_tridiagonal(a, b, c, rhs)

            u_new = [0.0] + u_new_inner + [0.0]
            u_new = self.apply_boundary(u_new, self.t[n + 1])

            U[n + 1] = u_new

        self.U = U
        return U

    def plot(self, times=[0.0, None]):
        if self.U is None:
            raise RuntimeError("Call solve() before plot().")

        plt.figure(figsize=(7, 4))

        for tau in times:
            if tau is None:
                idx = -1
            else:
                idx = min(range(len(self.t)), key=lambda i: abs(self.t[i] - tau))

            plt.plot(self.x, self.U[idx], label=f"t={self.t[idx]:.4f}")

        plt.title(f"Implicit Heat — λ = {self.lambda_cfl:.3f}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

class CrankNicolsonHeat1D:
    def __init__(self, L=1.0, Nx=100, T=0.1, Nt=100, alpha=1.0):
        self.L = L
        self.Nx = Nx
        self.T = T
        self.Nt = Nt
        self.alpha = alpha

        self.dx = L / Nx
        self.dt = T / Nt
        self.lambda_cfl = alpha * self.dt / self.dx**2

        self.x = [j * self.dx for j in range(Nx + 1)]
        self.t = [n * self.dt for n in range(Nt + 1)]

        self.U = None

    def initial_condition(self, x):
        return math.exp(-100 * (x - 0.5*self.L)**2)

    def apply_boundary(self, u, t):
        u[0] = 0.0
        u[-1] = 0.0
        return u

    def solve_tridiagonal(self, a, b, c, d):
        n = len(d)
        c_star = [0.0] * n
        d_star = [0.0] * n

        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_star[i-1]
            c_star[i] = c[i] / denom if i < n-1 else 0.0
            d_star[i] = (d[i] - a[i] * d_star[i-1]) / denom

        x = [0.0] * n
        x[-1] = d_star[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i+1]

        return x

    def solve(self):
        Nx, Nt = self.Nx, self.Nt
        lam = self.lambda_cfl

        U = [[0.0]*(Nx+1) for _ in range(Nt+1)]

        for j in range(Nx+1):
            U[0][j] = self.initial_condition(self.x[j])
        U[0] = self.apply_boundary(U[0], self.t[0])


        a = [-0.5*lam] * (Nx-1)
        b = [1.0 + lam] * (Nx-1)
        c = [-0.5*lam] * (Nx-1)

        for n in range(Nt):
            u_old = U[n]

            rhs = [0.0] * (Nx-1)
            for j in range(1, Nx):
                rhs[j-1] = (
                    0.5*lam * u_old[j-1]
                    + (1.0 - lam) * u_old[j]
                    + 0.5*lam * u_old[j+1]
                )

            u_new_inner = self.solve_tridiagonal(a, b, c, rhs)

            u_new = [0.0] + u_new_inner + [0.0]
            u_new = self.apply_boundary(u_new, self.t[n+1])

            U[n+1] = u_new

        self.U = U
        return U

    def plot(self, times=[0.0, None]):
        if self.U is None:
            raise RuntimeError("Call solve() before plot().")

        plt.figure(figsize=(7,4))

        for tau in times:
            if tau is None:
                idx = -1
            else:
                idx = min(range(len(self.t)), key=lambda i: abs(self.t[i] - tau))

            plt.plot(self.x, self.U[idx], label=f"t={self.t[idx]:.4f}")

        plt.title(f"Crank–Nicolson Heat — λ = {self.lambda_cfl:.3f}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()