import numpy as np


def compute_cholesky(matrix):
    """
    Cholesky decomposition: A = L * L^T
    Uses the Cholesky-Banachiewicz algorithm.

    For each entry:
      diagonal:     L[i,i] = sqrt( A[i,i] - sum_{k<i}( L[i,k]^2 ) )
      below diag:   L[i,j] = ( A[i,j] - sum_{k<j}( L[i,k]*L[j,k] ) ) / L[j,j]
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cholesky decomposition requires a square matrix.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Cholesky decomposition requires a symmetric matrix.")

    n = matrix.shape[0]
    L = np.zeros((n, n), dtype=float)
    steps = []

    steps.append({
        "desc": "Check: matrix is square and symmetric. Begin filling L row by row.",
        "formula": "A = L * L^T   (Cholesky-Banachiewicz)",
        "L": L.copy(),
    })

    for i in range(n):
        for j in range(i + 1):
            sum_k = float(sum(L[i, k] * L[j, k] for k in range(j)))

            if i == j:
                val = matrix[i, i] - sum_k
                if val <= 0:
                    raise ValueError(
                        f"Matrix is not positive definite at position ({i},{i}). "
                        f"Value under sqrt = {val:.6f}"
                    )
                L[i, j] = np.sqrt(val)
                steps.append({
                    "desc": f"Diagonal entry L[{i},{j}]",
                    "formula": (
                        f"L[{i},{i}] = sqrt( A[{i},{i}] - sum(L[{i},k]^2 for k<{i}) )\n"
                        f"         = sqrt( {matrix[i,i]:.4f} - {sum_k:.4f} )\n"
                        f"         = sqrt( {val:.4f} ) = {L[i,j]:.4f}"
                    ),
                    "highlight": (i, j),
                    "L": L.copy(),
                })
            else:
                L[i, j] = (matrix[i, j] - sum_k) / L[j, j]
                steps.append({
                    "desc": f"Off-diagonal entry L[{i},{j}]",
                    "formula": (
                        f"L[{i},{j}] = ( A[{i},{j}] - sum(L[{i},k]*L[{j},k] for k<{j}) ) / L[{j},{j}]\n"
                        f"         = ( {matrix[i,j]:.4f} - {sum_k:.4f} ) / {L[j,j]:.4f}\n"
                        f"         = {L[i,j]:.4f}"
                    ),
                    "highlight": (i, j),
                    "L": L.copy(),
                })

    return {
        "L (Lower Triangular)": np.round(L, 4),
        "L^T (Upper Triangular)": np.round(L.T, 4),
        "_steps": steps,
    }


def reconstruct_cholesky(L, LT):
    return L @ LT
