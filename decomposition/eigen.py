import numpy as np


def compute_eigen(matrix, max_iters=100):
    """
    Eigenvalue decomposition via QR iteration (Francis algorithm idea).
    A = V * Lambda * V^{-1}

    Each iteration:
      1. QR-decompose A_k  →  Q, R
      2. A_{k+1} = R * Q     (similar to A_k)
      3. Accumulate V = V * Q
    Diagonal of A_k converges to eigenvalues.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Eigenvalue decomposition requires a square matrix.")

    n = matrix.shape[0]
    A_k = matrix.astype(float).copy()
    V = np.eye(n, dtype=float)
    steps = []

    steps.append({
        "desc": "Start: A_0 = A,  V = I",
        "formula": "QR iteration: A_{k+1} = R_k * Q_k,  V = V * Q_k",
        "A_k": A_k.copy(),
        "V": V.copy(),
        "iteration": 0,
    })

    for it in range(max_iters):
        # Gram-Schmidt QR
        Q = np.zeros((n, n), dtype=float)
        R = np.zeros((n, n), dtype=float)
        for j in range(n):
            v = A_k[:, j].copy()
            for i in range(j):
                R[i, j] = float(np.dot(Q[:, i], A_k[:, j]))
                v = v - R[i, j] * Q[:, i]
            R[j, j] = float(np.linalg.norm(v))
            if R[j, j] > 1e-10:
                Q[:, j] = v / R[j, j]

        A_k = R @ Q
        V = V @ Q

        # Log every 10th iteration so we don't flood the UI
        if (it + 1) % max(1, max_iters // 5) == 0 or it == max_iters - 1:
            steps.append({
                "desc": f"After iteration {it+1}: diagonal approaches eigenvalues",
                "formula": (
                    f"A_{it+1} = R_{it} * Q_{it}\n"
                    f"V = V * Q_{it}\n"
                    f"Current diagonal: {np.round(np.diag(A_k), 4).tolist()}"
                ),
                "A_k": A_k.copy(),
                "V": V.copy(),
                "iteration": it + 1,
            })

    eigenvalues = np.diag(A_k)
    Lambda = np.diag(eigenvalues)

    steps.append({
        "desc": "Final: read eigenvalues from diagonal of converged A_k",
        "formula": "Lambda = diag(A_final),  A = V * Lambda * V^{-1}",
        "A_k": Lambda.copy(),
        "V": V.copy(),
        "iteration": "final",
    })

    return {
        "Eigenvalues (Lambda)": np.round(Lambda, 4),
        "Eigenvectors (V)": np.round(V, 4),
        "_steps": steps,
    }


def reconstruct_eigen(V, Lambda):
    try:
        V_inv = np.linalg.inv(V)
        return V @ Lambda @ V_inv
    except np.linalg.LinAlgError:
        raise ValueError("Eigenvector matrix is singular; reconstruction not possible.")
