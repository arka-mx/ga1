import numpy as np


class Eigen:
    def __init__(self):
        self.name = "Eigenvalue Decomposition"

    def evd(self, arr):
        matrix = np.array(arr, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("Eigenvalue decomposition requires a square matrix.")

        values, vectors = np.linalg.eig(matrix)
        return {
            "title": self.name,
            "components": [("Eigenvalues", values), ("Eigenvectors", vectors)],
            "message": "Computed eigenvalues and eigenvectors successfully.",
        }
