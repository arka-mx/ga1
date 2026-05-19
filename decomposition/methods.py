import numpy as np

from .cholesky import Cholesky
from .eigen import Eigen
from .lu import LU
from .qr import QR
from .svd import SVD


class Methods:
    def __init__(self):
        self.cholesky = Cholesky()
        self.eigen = Eigen()
        self.lu = LU()
        self.qr = QR()
        self.svd = SVD()

    def matrix_notes(self, arr):
        matrix = np.array(arr, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        rows, cols = matrix.shape
        notes = [f"Shape: {rows} x {cols}"]

        if rows == cols:
            rank = np.linalg.matrix_rank(matrix)
            determinant = float(np.linalg.det(matrix))
            notes.append(f"Rank: {rank}")
            notes.append(f"Determinant: {determinant:.6f}")
            if np.isclose(determinant, 0.0):
                notes.append("Matrix is singular.")
            else:
                notes.append("Matrix is non-singular.")
        else:
            rank = np.linalg.matrix_rank(matrix)
            notes.append(f"Rank: {rank}")
            notes.append("Matrix is not square, so determinant and singularity are not defined.")

        return notes
