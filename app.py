import numpy as np
import pandas as pd
import streamlit as st

from decomposition.lu import compute_lu, reconstruct_lu
from decomposition.qr import compute_qr, reconstruct_qr
from decomposition.svd import compute_svd, reconstruct_svd
from decomposition.eigen import compute_eigen, reconstruct_eigen
from decomposition.cholesky import compute_cholesky, reconstruct_cholesky
from decomposition.pca import compute_pca

from utils.validators import validate_matrix, dataframe_to_matrix, matrix_properties
from utils.helpers import (
    plot_heatmap,
    plot_heatmap_highlighted,
    plot_explained_variance,
    generate_random_matrix,
    time_execution,
)

st.set_page_config(
    page_title="Matrix Decomposition Studio",
    page_icon="🧮",
    layout="wide",
)

st.markdown("""
<style>

/* ── overall page ── */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Python-Tutor style step panel ── */
.tutor-step-header {
    background: #1e2a3a;
    color: #7ecfff;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.82rem;
    padding: 6px 14px;
    border-radius: 6px 6px 0 0;
    border: 1px solid #2e4060;
    border-bottom: none;
    letter-spacing: 0.05em;
}
.tutor-step-body {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.92rem;
    padding: 14px 18px;
    border: 1px solid #2e4060;
    border-top: none;
    border-radius: 0 0 0 0;
    white-space: pre-wrap;
    line-height: 1.7;
    min-height: 56px;
}
.tutor-formula-header {
    background: #2a2000;
    color: #f0c040;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.82rem;
    padding: 6px 14px;
    border: 1px solid #5a4500;
    border-bottom: none;
    letter-spacing: 0.05em;
}
.tutor-formula-body {
    background: #1a1400;
    color: #ffe082;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9rem;
    padding: 14px 18px;
    border: 1px solid #5a4500;
    border-top: none;
    border-radius: 0 0 6px 6px;
    white-space: pre-wrap;
    line-height: 1.8;
    min-height: 52px;
}

/* ── prop tags ── */
.prop-row { margin-top: 6px; }
.prop-tag {
    display: inline-block;
    background: #1e3050;
    color: #7ecfff;
    border-radius: 4px;
    padding: 2px 10px;
    margin: 3px 3px 0 0;
    font-size: 0.78rem;
    font-family: monospace;
}

/* ── step nav ── */
.step-nav-label {
    font-family: monospace;
    color: #888;
    font-size: 0.8rem;
    margin-bottom: 4px;
}

/* ── matrix label ── */
.matrix-label {
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.85rem;
    color: #7ecfff;
    background: #1e2a3a;
    padding: 3px 10px;
    border-radius: 4px 4px 0 0;
    display: inline-block;
    margin-bottom: -4px;
}

/* ── section divider ── */
.tutor-section {
    border-top: 1px solid #2e4060;
    margin: 18px 0 10px 0;
    padding-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ── helper to render a matrix table cleanly ──────────────────────────────────

def show_matrix(mat, label=None):
    arr = np.array(mat, dtype=float) if not np.iscomplexobj(mat) else np.array(mat)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if label:
        st.markdown(f'<div class="matrix-label">{label}</div>', unsafe_allow_html=True)
    if np.iscomplexobj(arr):
        st.dataframe(pd.DataFrame(arr).astype(str), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame(arr).style.format("{:.4f}"), use_container_width=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("1 · Matrix Input")

input_method = st.sidebar.radio(
    "How do you want to enter the matrix?",
    ["Manual Entry", "CSV Upload", "Random Generation"],
)

matrix = None

if input_method == "Manual Entry":
    rows = int(st.sidebar.number_input("Rows", 1, 10, 3))
    cols = int(st.sidebar.number_input("Columns", 1, 10, 3))
    default_df = pd.DataFrame(
        np.zeros((rows, cols)),
        columns=[f"C{c+1}" for c in range(cols)],
    )
    edited = st.sidebar.data_editor(default_df, num_rows="fixed", key="manual_editor")
    try:
        m = dataframe_to_matrix(edited)
        ok, msg, matrix = validate_matrix(m)
        if not ok:
            st.sidebar.error(msg)
    except ValueError as e:
        st.sidebar.error(str(e))

elif input_method == "CSV Upload":
    uploaded = st.sidebar.file_uploader("Upload CSV (no header row)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded, header=None)
            ok, msg, matrix = validate_matrix(df.values)
            if not ok:
                st.sidebar.error(msg)
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

elif input_method == "Random Generation":
    rand_rows = int(st.sidebar.number_input("Rows", 1, 20, 4, key="rr"))
    rand_cols = int(st.sidebar.number_input("Columns", 1, 20, 4, key="rc"))
    kind = st.sidebar.selectbox(
        "Matrix type",
        ["Random", "Square", "Symmetric Positive Definite"],
    )
    kind_map = {
        "Random": "random",
        "Square": "square",
        "Symmetric Positive Definite": "spd",
    }
    if st.sidebar.button("Generate"):
        st.session_state["rand_matrix"] = generate_random_matrix(
            rand_rows, rand_cols, kind_map[kind]
        )
    if "rand_matrix" in st.session_state:
        matrix = st.session_state["rand_matrix"]

st.sidebar.markdown("---")
st.sidebar.header("2 · Decomposition")

METHOD_NAMES = [
    "LU Decomposition",
    "QR Decomposition",
    "Eigenvalue Decomposition",
    "SVD",
    "Cholesky Decomposition",
    "PCA",
]
method = st.sidebar.selectbox("Choose method", METHOD_NAMES)

n_pca = 2
if method == "PCA":
    max_k = min(matrix.shape) if matrix is not None else 2
    n_pca = st.sidebar.slider("Principal components (k)", 1, max(max_k, 1), min(2, max_k))

run_btn = st.sidebar.button("▶  Compute", type="primary", use_container_width=True)


# ── page header ───────────────────────────────────────────────────────────────

st.title("🧮 Matrix Decomposition Studio")
st.markdown(
    "Enter a matrix, pick a decomposition, then walk through **every single "
    "calculation step** — the description, the formula, and the matrix state "
    "at that exact moment, just like Python Tutor."
)
st.markdown("---")

if matrix is None:
    st.info("👈  Use the sidebar to enter or generate a matrix.")
    st.stop()

# ── input matrix display ──────────────────────────────────────────────────────

st.subheader("Input Matrix")
col_mat, col_heat = st.columns([1, 1.5])
with col_mat:
    show_matrix(matrix)
    props = matrix_properties(matrix)
    tags = "".join(f'<span class="prop-tag">{p}</span>' for p in props)
    st.markdown(f'<div class="prop-row">{tags}</div>', unsafe_allow_html=True)
with col_heat:
    if st.checkbox("Show heatmap", value=True):
        st.plotly_chart(plot_heatmap(matrix, "Input Matrix"), use_container_width=True)

st.markdown("---")

# ── run decomposition ─────────────────────────────────────────────────────────

if run_btn:
    # clear any previous step position so slider starts at 0
    st.session_state.pop("current_step", None)

    result = None
    reconstruction = None
    error_msg = None
    exec_time = 0.0
    formula_summary = ""
    method_info = ""

    try:
        if method == "LU Decomposition":
            result, exec_time = time_execution(compute_lu, matrix)
            reconstruction = reconstruct_lu(result["L"], result["U"])
            formula_summary = "A = L  ×  U"
            method_info = (
                "LU splits a square matrix into lower triangular L and upper "
                "triangular U using Gaussian elimination. "
                "Each step computes one multiplier and eliminates one row."
            )

        elif method == "QR Decomposition":
            result, exec_time = time_execution(compute_qr, matrix)
            reconstruction = reconstruct_qr(result["Q"], result["R"])
            formula_summary = "A = Q  ×  R"
            method_info = (
                "QR uses the Gram-Schmidt process: each column of A is made "
                "orthogonal to all previous Q columns, then normalised. "
                "Q is orthogonal and R is upper triangular."
            )

        elif method == "Eigenvalue Decomposition":
            result, exec_time = time_execution(compute_eigen, matrix)
            reconstruction = reconstruct_eigen(
                result["Eigenvectors (V)"], result["Eigenvalues (Lambda)"]
            )
            formula_summary = "A = V  ×  Λ  ×  V⁻¹"
            method_info = (
                "Eigenvalue decomposition via QR iteration: repeatedly "
                "QR-decompose A_k and reassemble as R*Q. "
                "The diagonal converges to eigenvalues; accumulated Q columns become eigenvectors."
            )

        elif method == "SVD":
            result, exec_time = time_execution(compute_svd, matrix)
            reconstruction = reconstruct_svd(result["U"], result["Sigma"], result["V^T"])
            formula_summary = "A = U  ×  Σ  ×  Vᵀ"
            method_info = (
                "SVD works for any m×n matrix. "
                "Singular values are sqrt of eigenvalues of AᵀA. "
                "V comes from eigenvectors of AᵀA; U is computed as U[:,i] = A·v_i / σ_i."
            )

        elif method == "Cholesky Decomposition":
            result, exec_time = time_execution(compute_cholesky, matrix)
            reconstruction = reconstruct_cholesky(
                result["L (Lower Triangular)"], result["L^T (Upper Triangular)"]
            )
            formula_summary = "A = L  ×  Lᵀ"
            method_info = (
                "Cholesky requires the matrix to be symmetric and positive definite. "
                "The Banachiewicz algorithm fills L entry by entry: "
                "diagonal entries use a square root, off-diagonal use division."
            )

        elif method == "PCA":
            result, exec_time = time_execution(compute_pca, matrix, n_pca)
            formula_summary = "X_pca = X_centered  ×  W"
            method_info = (
                "PCA finds the directions of maximum variance. "
                "Steps: centre → covariance matrix → eigendecompose → pick top-k → project. "
                "Explained variance tells you how much information each component keeps."
            )

    except ValueError as ve:
        error_msg = str(ve)
    except Exception as ex:
        error_msg = f"Unexpected error: {ex}"

    if error_msg:
        st.error(f"**{method} failed:** {error_msg}")
        st.stop()

    # store everything in session state so the page doesn't re-run computation
    # when the user just moves the slider
    st.session_state["result"]        = result
    st.session_state["reconstruction"] = reconstruction
    st.session_state["exec_time"]     = exec_time
    st.session_state["formula_summary"] = formula_summary
    st.session_state["method_info"]   = method_info
    st.session_state["active_method"] = method
    st.session_state["current_step"]  = 0


# ── display results (only if we have stored data) ─────────────────────────────

if "result" not in st.session_state:
    st.info("Choose a method in the sidebar and click ▶ Compute.")
    st.stop()

result          = st.session_state["result"]
reconstruction  = st.session_state["reconstruction"]
exec_time       = st.session_state["exec_time"]
formula_summary = st.session_state["formula_summary"]
method_info     = st.session_state["method_info"]
active_method   = st.session_state["active_method"]
steps           = result.get("_steps", [])

SKIP = {"_steps", "_explained", "_n_components"}
output_items = [(k, v) for k, v in result.items() if k not in SKIP]

# ── method header ─────────────────────────────────────────────────────────────

st.subheader(f"Results — {active_method}")
st.info(method_info)
st.code(f"Formula:  {formula_summary}", language=None)
st.caption(f"Computed in {exec_time * 1000:.2f} ms")

# ── output matrices ───────────────────────────────────────────────────────────

st.markdown("#### Factorised Matrices")
num_cols = min(len(output_items), 4)
out_cols = st.columns(num_cols)
for i, (name, mat) in enumerate(output_items):
    with out_cols[i % num_cols]:
        show_matrix(mat, label=name)

# ── heatmaps toggle ───────────────────────────────────────────────────────────

if st.checkbox("Show heatmaps of output matrices"):
    h_cols = st.columns(min(len(output_items), 4))
    for i, (name, mat) in enumerate(output_items):
        arr = np.array(mat)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and not np.iscomplexobj(arr):
            with h_cols[i % len(h_cols)]:
                st.plotly_chart(plot_heatmap(arr, name), use_container_width=True)

# ── explained variance for PCA ────────────────────────────────────────────────

if active_method == "PCA" and "_explained" in result:
    st.markdown("#### Explained Variance")
    st.plotly_chart(plot_explained_variance(result["_explained"]), use_container_width=True)

# ── reconstruction check ──────────────────────────────────────────────────────

if reconstruction is not None:
    st.markdown("---")
    st.markdown("#### Verification  —  reconstructed A from the factors")
    ok = np.allclose(matrix, reconstruction, atol=1e-3)
    if ok:
        st.success("Reconstruction matches the original matrix (within numerical precision).")
    else:
        st.warning("Small numerical difference in reconstruction — expected for ill-conditioned matrices.")
    if st.checkbox("Show reconstructed matrix"):
        show_matrix(reconstruction, label="Reconstructed A")

# ── download ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Download Results")
dl_cols = st.columns(min(len(output_items), 4))
for i, (name, mat) in enumerate(output_items):
    arr = np.array(mat)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        if np.iscomplexobj(arr):
            csv = pd.DataFrame(arr).astype(str).to_csv(index=False, header=False)
        else:
            csv = pd.DataFrame(arr).to_csv(index=False, header=False)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("^", "")
        with dl_cols[i % len(dl_cols)]:
            st.download_button(
                label=f"⬇ {name}",
                data=csv,
                file_name=f"{safe_name}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON-TUTOR STYLE STEP WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔍 Step-by-Step Walkthrough")
st.markdown(
    "Move through each calculation step one at a time. "
    "Each step shows **what is happening**, **the exact formula**, "
    "and the **matrix state at that exact moment** with the changed cell highlighted."
)

if not steps:
    st.info("No step data available for this method.")
    st.stop()

total_steps = len(steps)

# ── navigation: use session state to track current step ──────────────────────
# We do NOT use st.slider as the source of truth because moving it reruns
# the whole app and the slider value resets. Instead we store the index in
# session_state and have Prev / Next buttons update it, plus a number_input
# that also drives it cleanly.

if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

def go_prev():
    if st.session_state["current_step"] > 0:
        st.session_state["current_step"] -= 1

def go_next():
    if st.session_state["current_step"] < total_steps - 1:
        st.session_state["current_step"] += 1

# navigation row
nav_left, nav_mid, nav_right = st.columns([1, 2, 1])

with nav_left:
    st.button("◀  Previous", on_click=go_prev, disabled=(st.session_state["current_step"] == 0))

with nav_mid:
    # number_input lets the user type a specific step number too
    chosen = st.number_input(
        f"Step (1 – {total_steps})",
        min_value=1,
        max_value=total_steps,
        value=st.session_state["current_step"] + 1,
        step=1,
        key="step_number",
    )
    st.session_state["current_step"] = int(chosen) - 1

with nav_right:
    st.button("Next  ▶", on_click=go_next, disabled=(st.session_state["current_step"] == total_steps - 1))

# progress bar
progress_pct = (st.session_state["current_step"]) / max(total_steps - 1, 1)
st.progress(progress_pct)

idx  = st.session_state["current_step"]
step = steps[idx]

# ── step info panel (Python-Tutor style dark code box) ───────────────────────

desc_text    = step.get("desc", "")
formula_text = step.get("formula", "")

st.markdown(
    f'<div class="tutor-step-header">▸  STEP {idx + 1} / {total_steps}</div>'
    f'<div class="tutor-step-body">{desc_text}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="tutor-formula-header">⟹  FORMULA</div>'
    f'<div class="tutor-formula-body">{formula_text}</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="tutor-section"></div>', unsafe_allow_html=True)

# ── matrices present in this step ────────────────────────────────────────────

# each key maps to a display label
STEP_MATRIX_KEYS = [
    ("L",      "L  (lower triangular)"),
    ("U",      "U  (upper triangular)"),
    ("Q",      "Q  (orthogonal)"),
    ("R",      "R  (upper triangular)"),
    ("A_k",    "A_k  (current iteration)"),
    ("V",      "V  (eigenvectors / principal components)"),
    ("matrix", step.get("label", "Matrix")),
]

highlight_cell = step.get("highlight")      # (row, col) tuple
highlight_col  = step.get("highlight_col")  # column index

to_render = []
for key, label in STEP_MATRIX_KEYS:
    if key in step:
        arr = np.array(step[key], dtype=float)
        to_render.append((label, arr))

if to_render:
    mat_cols = st.columns(min(len(to_render), 3))
    for i, (label, arr) in enumerate(to_render):
        with mat_cols[i % len(mat_cols)]:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            show_matrix(arr, label=label)
            # only show heatmap when matrix is 2-D and bigger than 1×1
            if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
                fig = plot_heatmap_highlighted(arr, label, highlight_cell, highlight_col)
                st.plotly_chart(fig, use_container_width=True)

# ── eigenvalues / extra info line ─────────────────────────────────────────────

if "eigenvalues" in step:
    ev = np.round(np.array(step["eigenvalues"]), 4)
    st.markdown(f"**Values at this step:** `{ev.tolist()}`")

if "extra" in step:
    st.info(step["extra"])
