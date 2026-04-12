import numpy as np
import matplotlib.pyplot as plt

class Newton:
    """
    Newton–Raphson root finder.
    """

    def __init__(self, function, derivative):
        self.f = function
        self.df = derivative
        self.history = None   # will store iterates after solve()

    def iteration(self, x):
        return x - self.f(x) / self.df(x)

    def solve(self, x0, tol=1e-8, max_iter=100):
        xs = [x0] # history includes x0 so plots show the full path from initial guess

        for _ in range(max_iter):
            x_old = xs[-1]
            x_new = self.iteration(x_old)


            if not np.isfinite \
                    (x_new): # Newton can diverge violently, derivative blows up near 0, iterates bounce and hit nan
                self.history = xs
                return {
                    "root": None,
                    "history": xs,
                    "reason": "Failed: iterate diverged (nan or inf)",
                    "iterations": len(xs),
                    "converged": False
                }
            done, reason = stop(x_new, x_old, self.f, tol)
            xs.append(x_new)

            if done:
                self.history = xs
                return {
                    "root": x_new,
                    "history": xs,
                    "reason": reason,
                    "iterations": len(xs)
                }

        self.history = xs
        return {
            "root": xs[-1],
            "history": xs,
            "reason": "Stopped: maximum iterations reached",
            "iterations": len(xs)
        }


    def plot(self, padding=1.0, show_tangents=True):
        if self.history is None:
            raise RuntimeError("Call solve() before plot().")

        xs = np.array(self.history)
        xmin, xmax = xs.min() - padding, xs.max() + padding

        # Smooth curve for f(x)
        X = np.linspace(xmin, xmax, 400)
        Y = self.f(X)

        plt.figure(figsize=(8, 5))
        plt.axhline(0, color='black', linewidth=1)

        # Plot the function
        plt.plot(X, Y, label="f(x)", color="blue")

        # Plot Newton iterates (except x0)
        plt.scatter(xs[1:], self.f(xs[1:]), color="red", zorder=5, label="Newton iterates")

        # Highlight the initial guess x0
        plt.scatter(xs[0], self.f(xs[0]), color="yellow", edgecolor="black",
                    s=120, zorder=6, label="Initial guess")


        # Vertical lines from x-axis to curve
        for x in xs:
            plt.plot([x, x], [0, self.f(x)], color="gray", linestyle="--", linewidth=0.8)

        if show_tangents:
            for x in xs[:3]:
                """
                Tangent lines for the first few iterations
                """
                y = self.f(x)
                slope = self.df(x)

                # tangent line: y_t = slope*(t - x) + y
                t = np.linspace(x - 1, x + 1, 50)
                y_t = slope * (t - x) + y

                plt.plot(t, y_t, color="green", alpha=0.2)


        plt.title("Newton's Method Convergence")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


class Secant:
    """
    Secant root finder.
    """

    def __init__(self, function):
        self.f = function
        self.history = None   # will store iterates after solve()

    def iteration(self, xn, xn_1):
        f_xn = self.f(xn)
        f_xn1 = self.f(xn_1)

        den = f_xn - f_xn1
        if abs(den) < 1e-14:
            raise ZeroDivisionError("Secant method denominator too small.")

        return xn - f_xn * ((xn - xn_1) / den)



    def solve(self, x1, x0, tol=1e-8, max_iter=100):
        xs = [x0, x1]

        for _ in range(max_iter):
            x_pold = xs[-2]
            x_old = xs[-1]
            x_new = self.iteration(x_old,x_pold)

            done, reason = stop(x_new, x_old, self.f, tol)
            xs.append(x_new)

            if done:
                self.history = xs
                return {
                    "root": x_new,
                    "history": xs,
                    "reason": reason,
                    "iterations": len(xs)
                }

        self.history = xs
        return {
            "root": xs[-1],
            "history": xs,
            "reason": "Stopped: maximum iterations reached",
            "iterations": len(xs)
        }


    def plot(self, padding=1.0, show_secants=True):
        if self.history is None:
            raise RuntimeError("Call solve() before plot().")

        xs = np.array(self.history)
        xmin, xmax = xs.min() - padding, xs.max() + padding

        # Smooth curve for f(x)
        X = np.linspace(xmin, xmax, 400)
        Y = self.f(X)

        plt.figure(figsize=(8, 5))
        plt.axhline(0, color='black', linewidth=1)

        # Plot the function
        plt.plot(X, Y, label="f(x)", color="blue")

        # Plot Secant iterates (except the very first)
        plt.scatter(xs[1:], self.f(xs[1:]), color="red", zorder=5, label="Secant iterates")

        # Highlight the initial guesses
        plt.scatter(xs[0], self.f(xs[0]), color="yellow", edgecolor="black",
                    s=120, zorder=6, label="Initial guess x0")
        plt.scatter(xs[1], self.f(xs[1]), color="orange", edgecolor="black",
                    s=120, zorder=6, label="Initial guess x1")

        # Vertical lines from x-axis to curve
        for x in xs:
            plt.plot([x, x], [0, self.f(x)], color="gray", linestyle="--", linewidth=0.8)

        # Draw secant lines between successive iterates
        if show_secants:
            for i in range(1, len(xs)-1):
                x0, x1 = xs[i-1], xs[i]
                y0, y1 = self.f(x0), self.f(x1)

                # Secant line between (x0, f(x0)) and (x1, f(x1))
                t = np.linspace(min(x0, x1)-1, max(x0, x1)+1, 50)
                # Equation of the secant line
                slope = (y1 - y0) / (x1 - x0)
                y_t = slope * (t - x1) + y1

                plt.plot(t, y_t, color="green", alpha=0.3)

        plt.title("Secant Method Convergence")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


class Bisection:
    """
    Bisection root finder.
    """

    def __init__(self, function):
        self.f = function
        self.history = None

    def solve(self, a, b, tol=1e-8, max_iter=100):
        fa = self.f(a)
        fb = self.f(b)

        if fa * fb > 0:
            raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")

        xs = []
        x_old = None

        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = self.f(m)
            xs.append(m)

            # stopping condition
            if x_old is not None:
                done, reason = stop(m, x_old, self.f, tol)
                if done:
                    self.history = xs
                    return {
                        "root": m,
                        "history": xs,
                        "reason": reason,
                        "iterations": len(xs)
                    }

            # update bracket
            if fa * fm < 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm

            x_old = m

        self.history = xs
        return {
            "root": xs[-1],
            "history": xs,
            "reason": "Stopped: maximum iterations reached",
            "iterations": len(xs)
        }

    def plot(self, padding=1.0):
        if self.history is None:
            raise RuntimeError("Call solve() before plot().")

        xs = np.array(self.history)
        xmin, xmax = xs.min() - padding, xs.max() + padding

        # Smooth curve for f(x)
        X = np.linspace(xmin, xmax, 400)
        Y = self.f(X)

        plt.figure(figsize=(8, 5))
        plt.axhline(0, color='black', linewidth=1)

        # Plot the function
        plt.plot(X, Y, label="f(x)", color="blue")

        # Plot midpoint iterates
        plt.scatter(xs, self.f(xs), color="red", zorder=5, label="Midpoints")

        # Vertical lines showing shrinking intervals
        for x in xs:
            plt.plot([x, x], [0, self.f(x)], color="gray", linestyle="--", linewidth=0.8)

        # Highlight first and last midpoint
        plt.scatter(xs[0], self.f(xs[0]), color="yellow", edgecolor="black",
                    s=120, zorder=6, label="First midpoint")
        plt.scatter(xs[-1], self.f(xs[-1]), color="orange", edgecolor="black",
                    s=120, zorder=6, label="Final midpoint")

        plt.title("Bisection Method Convergence")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

class Brent:
    """
    Brent–Dekker root finder.
    Hybrid of bisection, secant, and inverse quadratic interpolation.
    """

    def __init__(self, function):
        self.f = function
        self.history = None

    def solve(self, a, b, tol=1e-8, max_iter=100):
        fa = self.f(a)
        fb = self.f(b)

        if fa * fb > 0:
            raise ValueError("Brent requires f(a) and f(b) to have opposite signs.")

        # Initialization
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c, fc = a, fa
        d = e = b - a

        xs = []
        x_old = None

        for _ in range(max_iter):
            if fb == 0:
                self.history = xs
                return {
                    "root": b,
                    "history": xs,
                    "reason": "Exact root found",
                    "iterations": len(xs)
                }

            # Ensure b is best approximation
            if fa * fb > 0:
                a, fa = c, fc
                d = e = b - a

            if abs(fa) < abs(fb):
                c, b, a = b, a, c
                fc, fb, fa = fb, fa, fc

            # Convergence check
            m = 0.5 * (a - b)
            tol_act = 2 * tol * max(1.0, abs(b))

            xs.append(b)

            if abs(m) <= tol_act:
                self.history = xs
                return {
                    "root": b,
                    "history": xs,
                    "reason": "Stopped: tolerance achieved",
                    "iterations": len(xs)
                }

            # Choose interpolation or bisection
            if abs(e) >= tol_act and abs(fc) > abs(fb):
                # Attempt interpolation
                s = fb / fc
                if a == c:
                    # Secant
                    p = 2 * m * s
                    q = 1 - s
                else:
                    # Inverse quadratic interpolation
                    q = fc / fa
                    r = fb / fa
                    p = s * (2 * m * q * (q - r) - (b - c) * (r - 1))
                    q = (q - 1) * (r - 1) * (s - 1)

                if p > 0:
                    q = -q
                p = abs(p)

                # Accept interpolation only if safe
                if (2 * p < min(3 * m * q - abs(tol_act * q), abs(e * q))):
                    e, d = d, p / q
                else:
                    d = m
                    e = m
            else:
                # Bisection
                d = m
                e = m

            # Update
            c, fc = b, fb
            if abs(d) > tol_act:
                b += d
            else:
                b += tol_act if m > 0 else -tol_act

            fb = self.f(b)

        # Max iterations reached
        self.history = xs
        return {
            "root": b,
            "history": xs,
            "reason": "Stopped: maximum iterations reached",
            "iterations": len(xs)
        }

    def plot(self, padding=1.0):
        if self.history is None:
            raise RuntimeError("Call solve() before plot().")

        xs = np.array(self.history)
        xmin, xmax = xs.min() - padding, xs.max() + padding

        # Smooth curve for f(x)
        X = np.linspace(xmin, xmax, 400)
        Y = self.f(X)

        plt.figure(figsize=(8, 5))
        plt.axhline(0, color='black', linewidth=1)

        # Plot the function
        plt.plot(X, Y, label="f(x)", color="blue")

        # Plot Brent iterates
        plt.scatter(xs, self.f(xs), color="red", zorder=5, label="Brent iterates")

        # Vertical lines from x-axis to curve
        for x in xs:
            plt.plot([x, x], [0, self.f(x)], color="gray", linestyle="--", linewidth=0.8)

        # Highlight first and last iterate
        plt.scatter(xs[0], self.f(xs[0]), color="yellow", edgecolor="black",
                    s=120, zorder=6, label="First iterate")
        plt.scatter(xs[-1], self.f(xs[-1]), color="orange", edgecolor="black",
                    s=120, zorder=6, label="Final iterate")

        plt.title("Brent's Method Convergence")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()