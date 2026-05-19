import numpy as np
import pandas as pd
import streamlit as st

from decomposition.methods import Methods


st.set_page_config(
    page_title="Matrix Decomposition Studio",
    page_icon="M",
    layout="wide",
)


def build_default_dataframe(rows, cols):
    return pd.DataFrame(
        np.zeros((rows, cols), dtype=float),
        columns=[f"C{index + 1}" for index in range(cols)],
    )


def dataframe_to_matrix(dataframe):
    numeric = dataframe.apply(pd.to_numeric, errors="coerce")
    if numeric.isnull().values.any():
        raise ValueError("Every matrix cell must contain a valid number.")
    return numeric.to_numpy(dtype=float)


def render_component(label, value):
    st.markdown(f"#### {label}")
    array = np.array(value)

    if array.ndim == 1:
        st.dataframe(
            pd.DataFrame(array.reshape(1, -1), index=[label]),
            use_container_width=True,
        )
    else:
        st.dataframe(pd.DataFrame(array), use_container_width=True)


methods = Methods()

decomposition_options = {
    "Cholesky": methods.cholesky.cd,
    "Eigenvalue": methods.eigen.evd,
    "LU": methods.lu.lu,
    "QR": methods.qr.qr,
    "SVD": methods.svd.svd,
}


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(16, 185, 129, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }
    .hero {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .note-card {
        background: rgba(255, 255, 255, 0.72);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.35rem;">Matrix Decomposition Studio</h1>
        <p style="margin:0;color:#334155;">
            Enter a matrix, choose a decomposition, and inspect the factorized matrices with clear labels and matrix checks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1.35], gap="large")

with left_col:
    st.subheader("Matrix Setup")
    row_count = st.number_input("Number of rows", min_value=1, max_value=10, value=3, step=1)
    col_count = st.number_input("Number of columns", min_value=1, max_value=10, value=3, step=1)
    decomposition_name = st.selectbox(
        "Type of decomposition",
        list(decomposition_options.keys()),
    )

    editor_key = f"matrix-editor-{row_count}-{col_count}"
    matrix_df = st.data_editor(
        build_default_dataframe(int(row_count), int(col_count)),
        key=editor_key,
        num_rows="fixed",
        use_container_width=True,
    )

    decompose_clicked = st.button("Decompose", type="primary", use_container_width=True)

with right_col:
    st.subheader("Result")
    st.markdown(
        '<div class="note-card">Choose dimensions, fill the matrix, and click <strong>Decompose</strong> to see the result here.</div>',
        unsafe_allow_html=True,
    )

if decompose_clicked:
    try:
        matrix = dataframe_to_matrix(matrix_df)
        notes = methods.matrix_notes(matrix)
        result = decomposition_options[decomposition_name](matrix)

        with right_col:
            st.subheader(result["title"])
            st.success(result["message"])

            st.markdown("#### Input Matrix")
            st.dataframe(pd.DataFrame(matrix), use_container_width=True)

            st.markdown("#### Matrix Checks")
            for note in notes:
                st.info(note)

            for label, value in result["components"]:
                render_component(label, value)

            if decomposition_name in {"LU", "Eigenvalue", "Cholesky"}:
                if matrix.shape[0] == matrix.shape[1] and np.isclose(np.linalg.det(matrix), 0.0):
                    st.warning(
                        "This square matrix is singular. Some decompositions can still be computed, but the matrix is not invertible."
                    )

    except ValueError as exc:
        with right_col:
            st.error(str(exc))
    except Exception as exc:
        with right_col:
            st.error(f"Unable to decompose the matrix: {exc}")
