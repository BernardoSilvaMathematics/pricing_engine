import numpy as np
import time
import matplotlib.pyplot as plt


class LU_nopivot:
    """
    Out-of-place LU decomposition without pivoting.
    Computes A = L U where:
        - L is unit lower triangular (diagonal = 1)
        - U is upper triangular
    """

    def __init__(self, A):
        A = np.array(A, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("LU decomposition requires a square matrix.")

        self.A = A
        self.n = A.shape[0]
        self.L = None
        self.U = None

    def factor(self):
        """
        Compute L and U out-of-place.
        Raises an error if a zero pivot is encountered.
        """
        n = self.n
        A = self.A

        L = np.eye(n)
        U = A.copy()

        for k in range(n):
            pivot = U[k, k]

            # Existence check - since this is a no pivot
            if abs(pivot) < 1e-14:
                raise ValueError(f"Zero pivot encountered at index {k}. "
                                 "LU without pivoting does not exist.")

            for i in range(k + 1, n):
                L[i, k] = U[i, k] / pivot

                for j in range(k, n):
                    U[i, j] -= L[i, k] * U[k, j]

        self.L = L
        self.U = U
        return L, U

    def forward_sub(self, b):
        n = self.n
        y = np.zeros(n)

        for i in range(n):
            y[i] = b[i] - np.dot(self.L[i, :i], y[:i])
        return y

    def backward_sub(self, y):
        n = self.n
        x = np.zeros(n)

        for i in reversed(range(n)):
            if abs(self.U[i, i]) < 1e-14:
                raise ValueError("Zero diagonal entry in U during back substitution.")
            x[i] = (y[i] - np.dot(self.U[i, i + 1:], x[i + 1:])) / self.U[i, i]
        return x

    def solve(self, b):
        if self.L is None or self.U is None:
            self.factor()

        b = np.array(b, dtype=float)

        y = self.forward_sub(b)
        x = self.backward_sub(y)
        return x


class LU_optimized:
    def __init__(self, A):
        A = np.array(A, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("LU decomposition requires a square matrix.")

        self.A = A
        self.n = A.shape[0]
        self.piv = np.arange(self.n)  # permutation vector
        self._factored = False

    def factor(self):
        self._factored = True
        A = self.A
        n = self.n
        piv = self.piv

        for k in range(n):
            p = k + np.argmax(np.abs(A[k:, k]))
            if abs(A[p, k]) < 1e-14:
                raise ValueError("Matrix is singular to working precision.")

            if p != k:
                A[[k, p], :] = A[[p, k], :]
                piv[[k, p]] = piv[[p, k]]

            A[k + 1:, k] /= A[k, k]
            A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])

        return A, piv

    def solve(self, b):
        if not self._factored:
            self.factor()

        A = self.A
        piv = self.piv
        n = self.n

        b = np.array(b, dtype=float)
        b = b[piv]

        for i in range(n):
            b[i + 1:] -= A[i + 1:, i] * b[i]

        for i in reversed(range(n)):
            b[i] /= A[i, i]
            b[:i] -= A[:i, i] * b[i]

        return b

def is_symmetric(A, tol=1e-12):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True

class Cholesky:
    def __init__(self, A):
        A = np.array(A, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Cholesky decomposition requires a square matrix.")

        if not is_symmetric(A):
            raise ValueError("Matrix must be symmetric.")

        self.A = A
        self.n = A.shape[0]
        self._factored = False

    def factor(self):
        A = self.A
        n = self.n

        self._factored = True

        for k in range(n):
            s = 0.0
            for j in range(k):
                s += A[k][j] * A[k][j]

            val = A[k][k] - s
            if val <= 0:
                raise ValueError("Matrix is not positive definite.")
            A[k][k] = val ** 0.5

            # column below
            for i in range(k + 1, n):
                s = 0.0
                for j in range(k):
                    s += A[i][j] * A[k][j]

                A[i][k] = (A[i][k] - s) / A[k][k]
        for i in range(n):
            for j in range(i + 1, n):
                A[i][j] = 0.0

        return A

    def solve(self, b):
        if not self._factored:
            self.factor()

        A = self.A
        n = self.n
        b = list(map(float, b))

        for i in range(n):
            s = 0.0
            for j in range(i):
                s += A[i][j] * b[j]
            b[i] = (b[i] - s) / A[i][i]

        for i in reversed(range(n)):
            s = 0.0
            for j in range(i + 1, n):
                s += A[j][i] * b[j]
            b[i] = (b[i] - s) / A[i][i]

        return b