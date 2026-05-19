import numpy as np


class Cholesky:
    def __init__(self):
        self.name = "Cholesky Decomposition"

    def cd(self, arr):
        matrix = np.array(arr, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("Cholesky decomposition requires a square matrix.")

        if not np.allclose(matrix, matrix.T):
            raise ValueError("Cholesky decomposition requires a symmetric matrix.")

        eigenvalues = np.linalg.eigvalsh(matrix)
        if not np.all(eigenvalues > 0):
            raise ValueError(
                "Cholesky decomposition requires a positive definite matrix."
            )

        lower = np.linalg.cholesky(matrix)
        return {
            "title": self.name,
            "components": [("L", lower), ("L^T", lower.T)],
            "message": "Matrix is symmetric positive definite.",
        }
