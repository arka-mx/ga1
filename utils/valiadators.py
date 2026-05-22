import numpy as np
import pandas as pd


METHOD_REQUIREMENTS = {
    "Cholesky": (
        "Square matrix",
        "Symmetric matrix",
        "Positive definite matrix",
    ),
    "Eigenvalue": ("Square matrix",),
    "LU": ("Square matrix",),
    "QR": ("Any numeric matrix",),
    "SVD": ("Any numeric matrix",),
}


def build_default_dataframe(rows, cols):
    return pd.DataFrame(
        np.zeros((rows, cols), dtype=float),
        columns=[f"C{index + 1}" for index in range(cols)],
    )


def dataframe_to_matrix(dataframe):
    numeric = dataframe.apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().values.any():
        raise ValueError("Every matrix cell must contain a valid number.")

    matrix = numeric.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix values must be finite numbers.")

    return matrix


def _is_square(matrix):
    return matrix.shape[0] == matrix.shape[1]


def _is_symmetric(matrix):
    return _is_square(matrix) and np.allclose(matrix, matrix.T)


def _is_positive_definite(matrix):
    if not _is_symmetric(matrix):
        return False

    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False

    return True


def summarize_matrix(matrix):
    rows, cols = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    summary = {
        "rows": rows,
        "cols": cols,
        "shape": f"{rows} x {cols}",
        "rank": rank,
        "is_square": _is_square(matrix),
        "is_symmetric": _is_symmetric(matrix),
        "is_positive_definite": _is_positive_definite(matrix),
        "has_finite_values": bool(np.isfinite(matrix).all()),
    }

    if summary["is_square"]:
        determinant = float(np.linalg.det(matrix))
        summary["determinant"] = determinant
        summary["is_singular"] = bool(np.isclose(determinant, 0.0))
    else:
        summary["determinant"] = None
        summary["is_singular"] = None

    return summary


def matrix_notes(matrix):
    summary = summarize_matrix(matrix)
    notes = [
        f"Shape: {summary['shape']}",
        f"Rank: {summary['rank']}",
    ]

    if summary["is_square"]:
        notes.append(f"Determinant: {summary['determinant']:.6f}")
        notes.append(
            "Matrix is singular."
            if summary["is_singular"]
            else "Matrix is non-singular."
        )
    else:
        notes.append(
            "Matrix is not square, so determinant and singularity are not defined."
        )

    return notes


def validate_for_method(matrix, method_name):
    summary = summarize_matrix(matrix)
    errors = []
    warnings = []

    if not summary["has_finite_values"]:
        errors.append("Matrix values must be finite numbers.")

    if method_name in {"Cholesky", "Eigenvalue", "LU"} and not summary["is_square"]:
        errors.append(f"{method_name} decomposition requires a square matrix.")

    if method_name == "Cholesky":
        if summary["is_square"] and not summary["is_symmetric"]:
            errors.append("Cholesky decomposition requires a symmetric matrix.")
        if summary["is_square"] and summary["is_symmetric"] and not summary["is_positive_definite"]:
            errors.append(
                "Cholesky decomposition requires a positive definite matrix."
            )

    if method_name in {"LU", "Eigenvalue", "Cholesky"} and summary["is_square"]:
        if summary["is_singular"]:
            warnings.append(
                "This square matrix is singular, so it is not invertible."
            )

    return {
        "summary": summary,
        "requirements": METHOD_REQUIREMENTS[method_name],
        "errors": errors,
        "warnings": warnings,
        "is_valid": not errors,
    }
