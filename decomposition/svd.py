import numpy as np


def compute_svd(matrix):
    """
    Singular Value Decomposition:  A = U * Sigma * V^T

    Steps:
      1. Compute A^T * A
      2. Eigendecompose A^T A  →  eigenvalues λ_i, eigenvectors V
      3. Singular values σ_i = sqrt(λ_i)
      4. Build U: u_i = (A * v_i) / σ_i
      5. Assemble Sigma (diagonal matrix of σ_i)
    """
    A = matrix.astype(float).copy()
    n, m = A.shape
    steps = []

    # Step 1
    AtA = A.T @ A
    steps.append({
        "desc": "Step 1 – Compute A^T * A",
        "formula": "A^T A = A transposed multiplied by A",
        "matrix": AtA.copy(),
        "label": "A^T A",
    })

    # Step 2
    eigenvalues, V = np.linalg.eig(AtA)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx].real
    V = V[:, idx].real
    steps.append({
        "desc": "Step 2 – Eigendecompose A^T A (sort descending)",
        "formula": "A^T A · v_i = λ_i · v_i   →  columns of V are right singular vectors",
        "matrix": V.copy(),
        "label": "V (right singular vectors)",
        "eigenvalues": eigenvalues.copy(),
    })

    # Step 3
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    steps.append({
        "desc": "Step 3 – Compute singular values",
        "formula": "σ_i = sqrt(λ_i)",
        "matrix": np.diag(singular_values),
        "label": "Singular values (diagonal)",
        "eigenvalues": singular_values.copy(),
    })

    # Step 4
    U = np.zeros((n, n), dtype=float)
    for i, sigma in enumerate(singular_values):
        if i < m and sigma > 1e-10:
            U[:, i] = (A @ V[:, i]) / sigma
    # fill remaining columns with orthonormal basis if needed
    steps.append({
        "desc": "Step 4 – Compute U via u_i = A·v_i / σ_i",
        "formula": "u_i = (A * v_i) / σ_i   (left singular vectors)",
        "matrix": U.copy(),
        "label": "U (left singular vectors)",
    })

    k = min(n, m)
    U_out = U[:, :k]
    S = singular_values[:k]
    Sigma = np.diag(S)
    VT = V.T[:k, :]

    steps.append({
        "desc": "Step 5 – Assemble full decomposition A = U Σ V^T",
        "formula": "A = U * Sigma * V^T",
        "matrix": Sigma.copy(),
        "label": "Sigma",
    })

    return {
        "U": np.round(U_out, 4),
        "Sigma": np.round(Sigma, 4),
        "V^T": np.round(VT, 4),
        "_steps": steps,
    }


def reconstruct_svd(U, Sigma, VT):
    return U @ Sigma @ VT
