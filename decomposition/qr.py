import numpy as np


def compute_qr(matrix):
    """
    QR decomposition via the classical Gram-Schmidt process.

    Formula:
      v_j = a_j - sum_{i<j}( (q_i · a_j) * q_i )   (projection removal)
      R[i,j] = q_i · a_j                              (projection coefficient)
      R[j,j] = ||v_j||                                (column norm)
      q_j    = v_j / R[j,j]                           (normalise)
    """
    A = matrix.astype(float).copy()
    n, m = A.shape

    Q = np.zeros((n, m), dtype=float)
    R = np.zeros((m, m), dtype=float)
    steps = []

    steps.append({
        "desc": "Start: columns of A will be orthogonalised one by one",
        "formula": "Q = zeros, R = zeros",
        "Q": Q.copy(),
        "R": R.copy(),
    })

    for j in range(m):
        v = A[:, j].copy()
        step_projections = []

        for i in range(j):
            R[i, j] = float(np.dot(Q[:, i], A[:, j]))
            proj = R[i, j] * Q[:, i]
            step_projections.append(
                f"proj onto q_{i}: R[{i},{j}] = q_{i}·a_{j} = {R[i,j]:.4f}"
            )
            v = v - proj

        R[j, j] = float(np.linalg.norm(v))
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = 0.0

        proj_text = "\n".join(step_projections) if step_projections else "No prior columns yet."
        steps.append({
            "desc": f"Orthogonalise column {j}",
            "formula": (
                f"{proj_text}\n"
                f"v = a_{j} minus projections\n"
                f"R[{j},{j}] = ||v|| = {R[j,j]:.4f}\n"
                f"q_{j} = v / {R[j,j]:.4f}"
            ),
            "highlight_col": j,
            "Q": Q.copy(),
            "R": R.copy(),
        })

    return {
        "Q": np.round(Q, 4),
        "R": np.round(R, 4),
        "_steps": steps,
    }


def reconstruct_qr(Q, R):
    return Q @ R
