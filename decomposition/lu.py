import numpy as np
from scipy.linalg import lu


class LU:
    def __init__(self):
        self.name = "LU Decomposition"

    def lu(self, arr):
        matrix = np.array(arr, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("LU decomposition requires a square matrix.")

        permutation, lower, upper = lu(matrix)
        return {
            "title": self.name,
            "components": [("P", permutation), ("L", lower), ("U", upper)],
            "message": "Computed permutation, lower, and upper triangular matrices.",
        }
