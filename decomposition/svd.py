import numpy as np


class SVD:
    def __init__(self):
        self.name = "Singular Value Decomposition"

    def svd(self, arr):
        matrix = np.array(arr, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        left, singular_values, right_t = np.linalg.svd(matrix)
        sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
        np.fill_diagonal(sigma, singular_values)
        return {
            "title": self.name,
            "components": [("U", left), ("Sigma", sigma), ("V^T", right_t)],
            "message": "Computed singular values and orthogonal factors.",
        }
