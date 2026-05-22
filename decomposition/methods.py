import numpy as np

from .cholesky import Cholesky
from .eigen import Eigen
from .lu import LU
from .qr import QR
from .svd import SVD
from utils.valiadators import matrix_notes


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
        return matrix_notes(matrix)
