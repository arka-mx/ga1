import numpy as np
import pandas as pd


def validate_matrix(data):
    """
    Convert input (DataFrame values, list, or ndarray) to a clean float ndarray.
    Returns (is_valid, error_message, matrix_or_None).
    """
    try:
        matrix = np.array(data, dtype=float)
    except Exception as exc:
        return False, f"Cannot convert to numeric matrix: {exc}", None

    if matrix.ndim != 2:
        return False, "Input must be a 2-D matrix.", None
    if matrix.size == 0:
        return False, "Matrix cannot be empty.", None
    if np.isnan(matrix).any():
        return False, "Matrix contains NaN values. Fill every cell with a number.", None
    if np.isinf(matrix).any():
        return False, "Matrix contains Inf values. Use finite numbers only.", None

    return True, "OK", matrix


def dataframe_to_matrix(df):
    """
    Convert a Streamlit data_editor DataFrame to a numpy float array.
    Raises ValueError if any cell is non-numeric.
    """
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().values.any():
        raise ValueError("Every matrix cell must contain a valid number.")
    return numeric.to_numpy(dtype=float)


def is_square(matrix):
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]


def is_symmetric(matrix):
    return is_square(matrix) and np.allclose(matrix, matrix.T)


def is_positive_definite(matrix):
    if not is_symmetric(matrix):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > 0))
    except np.linalg.LinAlgError:
        return False


def matrix_properties(matrix):
    """Return a list of human-readable property strings about the matrix."""
    rows, cols = matrix.shape
    notes = [f"Shape: {rows} × {cols}"]

    if is_square(matrix):
        rank = int(np.linalg.matrix_rank(matrix))
        det = float(np.linalg.det(matrix))
        notes.append(f"Rank: {rank}")
        notes.append(f"Determinant: {det:.6f}")
        notes.append("Singular" if np.isclose(det, 0.0) else "Non-singular")
        if is_symmetric(matrix):
            notes.append("Symmetric")
        if is_positive_definite(matrix):
            notes.append("Positive Definite")
    else:
        rank = int(np.linalg.matrix_rank(matrix))
        notes.append(f"Rank: {rank}")
        notes.append("Non-square – det / singularity not defined")

    return notes
