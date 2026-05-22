import numpy as np


def compute_lu(matrix):
    """
    LU decomposition without pivoting using Gaussian elimination.
    Returns L and U matrices plus a step-by-step log for visualization.

    Formula:  A = L * U
      - L[j,i] = U[j,i] / U[i,i]          (multiplier)
      - U[j]   = U[j] - L[j,i] * U[i]     (row elimination)
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("LU decomposition requires a square matrix.")

    n = matrix.shape[0]
    U = matrix.astype(float).copy()
    L = np.eye(n, dtype=float)
    steps = []

    steps.append({
        "desc": "Start: copy A into U, set L = identity",
        "formula": "U = A,  L = I",
        "U": U.copy(),
        "L": L.copy(),
    })

    for i in range(n):
        pivot = U[i, i]
        if abs(pivot) < 1e-12:
            raise ValueError(
                f"Zero pivot at position ({i},{i}). "
                "Try a different matrix or use partial pivoting."
            )
        for j in range(i + 1, n):
            multiplier = U[j, i] / pivot
            L[j, i] = multiplier
            U[j] = U[j] - multiplier * U[i]
            steps.append({
                "desc": f"Eliminate column {i}, row {j}",
                "formula": (
                    f"L[{j},{i}] = U[{j},{i}] / U[{i},{i}] = "
                    f"{U[j,i]+multiplier*U[i,i]:.4f} / {pivot:.4f} = {multiplier:.4f}\n"
                    f"U[{j},:] = U[{j},:] - {multiplier:.4f} × U[{i},:]"
                ),
                "highlight": (j, i),
                "U": U.copy(),
                "L": L.copy(),
            })

    return {
        "L": np.round(L, 4),
        "U": np.round(U, 4),
        "_steps": steps,
    }


def reconstruct_lu(L, U):
    return L @ U
