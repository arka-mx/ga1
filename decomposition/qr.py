import numpy as np


class QR:
    def __init__(self):
        self.name = "QR Decomposition"

    def qr(self, arr):
        matrix = np.array(arr, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        orthogonal, upper = np.linalg.qr(matrix)
        return {
            "title": self.name,
            "components": [("Q", orthogonal), ("R", upper)],
            "message": "Computed orthogonal and upper triangular matrices.",
        }
