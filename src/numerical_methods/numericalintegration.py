import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Trapezoid:
    """
    n_evals : Number of function evaluations performed during the last call to integrate().
    """

    def __init__(self):
        self.n_evals = 0

    def integrate(self, f, a, b, n):
        h = (b - a) / n
        self.n_evals = 0

        s = 0.5 * (f(a) + f(b))
        self.n_evals += 2

        for k in range(1, n):
            s += f(a + k * h)
            self.n_evals += 1

        return h * s

    def error(self, f, a, b, n, exact):
        approx = self.integrate(f, a, b, n)
        err = abs(approx - exact)
        return approx, err, self.n_evals


class Simpson:
    def __init__(self):
        self.n_evals = 0

    def integrate(self, f, a, b, n):
        if n % 2 != 0:
            raise ValueError("Simpson's rule requires the number of subintervals n to be even.")

        h = (b - a) / n
        self.n_evals = 0

        s = f(a) + f(b)
        self.n_evals += 2

        for k in range(1, n):
            xk = a + k * h
            if k % 2 == 1:
                s += 4 * f(xk)
            else:
                s += 2 * f(xk)
            self.n_evals += 1

        return (h / 3) * s

    def error(self, f, a, b, n, exact):
        approx = self.integrate(f, a, b, n)
        err = abs(approx - exact)
        return approx, err, self.n_evals

def legendre(n, x):
    """
    Finding the roots so we can later use as nodes on the quadrature
    """
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0

    Pnm2 = 1.0
    Pnm1 = x
    dPnm2 = 0.0
    dPnm1 = 1.0

    for k in range(2, n + 1):
        Pn = ((2*k - 1)*x*Pnm1 - (k - 1)*Pnm2) / k
        dPn = dPnm2 + (2*k - 1)*Pnm1

        Pnm2, Pnm1 = Pnm1, Pn
        dPnm2, dPnm1 = dPnm1, dPn

    return Pn, dPn


class GaussLegendre:

    def __init__(self, n, tol=1e-14, max_iter=50):
        self.n = n
        self.tol = tol
        self.max_iter = max_iter
        self.nodes, self.weights = self._compute_nodes_weights()
        self.n_evals = 0

    def _compute_nodes_weights(self):
        n = self.n
        nodes = np.zeros(n)
        weights = np.zeros(n)

        if n < 1:
            raise ValueError("Number of quadrature points n must be >= 1.")

        # Not theoretically necessary but numerical cap
        # As the Legendre polynomials are computed via three theme recurrence (root clusterinf and roundoff amplification)
        # This is simple and efficient for moderante n, but if n grows too large it becomes unstable
        # Famous libraries switch to Golub Welsch method or use extended precision to avoid these issues
        # Methods such as the Golub-Welsch compute the nodes as eigen values of a symmetric tridiagonal Jacobi matrix
        if n > 50:
            raise ValueError(
                "Number of quadrature points n too large for stable Newton-based Gauss–Legendre computation.")

        m = (n + 1) // 2

        for i in range(m):
            # asymptotic behavior of Legendre roots
            x = np.cos(np.pi * (i + 0.75) / (n + 0.5))

            # Newton iterations
            for _ in range(self.max_iter):
                Pn, dPn = legendre(n, x)
                dx = -Pn / dPn
                x += dx
                if abs(dx) < self.tol:
                    break

            nodes[i] = x
            nodes[-(i + 1)] = -x

            weights[i] = 2 / ((1 - x ** 2) * (dPn ** 2))
            weights[-(i + 1)] = weights[i]

        return nodes, weights

    def integrate(self, f, a, b):
        self.n_evals = 0

        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)

        total = 0.0
        for xk, wk in zip(self.nodes, self.weights):
            total += wk * f(mid + half * xk)
            self.n_evals += 1

        return half * total

    def error(self, f, a, b, exact):
        approx = self.integrate(f, a, b)
        err = abs(approx - exact)
        return approx, err, self.n_evals


class AdaptiveSimpson:
    def __init__(self, tol=1e-8, max_depth=20):
        self.tol = tol
        self.max_depth = max_depth
        self.n_evals = 0

    def _simpson(self, a, b, fa, fm, fb):
        return (b - a) * (fa + 4*fm + fb) / 6

    def _adaptive(self, f, a, b, fa, fm, fb, S, tol, depth):
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)

        flm = f(lm)
        frm = f(rm)
        self.n_evals += 2

        S_left  = self._simpson(a, m, fa, flm, fm)
        S_right = self._simpson(m, b, fm, frm, fb)
        S2 = S_left + S_right

        if abs(S2 - S) < 15 * tol or depth <= 0:
            # Richardson extrapolation correction
            return S2 + (S2 - S) / 15
        else:
            left  = self._adaptive(f, a, m, fa, flm, fm, S_left,  tol/2, depth-1)
            right = self._adaptive(f, m, b, fm, frm, fb, S_right, tol/2, depth-1)
            return left + right

    def integrate(self, f, a, b):
        self.n_evals = 0

        fa = f(a)
        fb = f(b)
        m  = 0.5 * (a + b)
        fm = f(m)
        self.n_evals += 3

        S = self._simpson(a, b, fa, fm, fb)

        return self._adaptive(f, a, b, fa, fm, fb, S, self.tol, self.max_depth)

    def error(self, f, a, b, exact):
        approx = self.integrate(f, a, b)
        err = abs(approx - exact)
        return approx, err, self.n_evals