import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from decomposition.methods import Methods
from utils.valiadators import (
    METHOD_REQUIREMENTS,
    build_default_dataframe,
    dataframe_to_matrix,
    summarize_matrix,
    validate_for_method,
)


st.set_page_config(
    page_title="Matrix Decomposition Studio",
    page_icon="M",
    layout="wide",
)


methods = Methods()

DECOMPOSITION_OPTIONS = {
    "Cholesky": {
        "fn": methods.cholesky.cd,
        "subtitle": "For symmetric positive definite matrices.",
    },
    "Eigenvalue": {
        "fn": methods.eigen.evd,
        "subtitle": "Breaks a square matrix into eigenvalues and eigenvectors.",
    },
    "LU": {
        "fn": methods.lu.lu,
        "subtitle": "Splits a square matrix into permutation, lower, and upper factors.",
    },
    "QR": {
        "fn": methods.qr.qr,
        "subtitle": "Produces an orthogonal matrix and an upper triangular matrix.",
    },
    "SVD": {
        "fn": methods.svd.svd,
        "subtitle": "Works for any numeric matrix and exposes its singular values.",
    },
}


def init_session_state():
    if "matrix_df" not in st.session_state:
        st.session_state.matrix_df = build_default_dataframe(3, 3)
    if "matrix_shape" not in st.session_state:
        st.session_state.matrix_shape = (3, 3)
    if "result_payload" not in st.session_state:
        st.session_state.result_payload = None


def resize_dataframe(dataframe, rows, cols):
    resized = build_default_dataframe(rows, cols)
    overlap_rows = min(rows, dataframe.shape[0])
    overlap_cols = min(cols, dataframe.shape[1])

    if overlap_rows and overlap_cols:
        resized.iloc[:overlap_rows, :overlap_cols] = dataframe.iloc[
            :overlap_rows, :overlap_cols
        ].to_numpy()

    return resized


def generate_random_dataframe(rows, cols, min_value, max_value, integers_only):
    if integers_only:
        values = np.random.randint(min_value, max_value + 1, size=(rows, cols))
    else:
        values = np.random.uniform(min_value, max_value, size=(rows, cols))
        values = np.round(values, 4)

    return pd.DataFrame(
        values,
        columns=[f"C{index + 1}" for index in range(cols)],
    )


def sync_matrix_shape(rows, cols):
    target_shape = (rows, cols)
    if st.session_state.matrix_shape != target_shape:
        st.session_state.matrix_df = resize_dataframe(
            st.session_state.matrix_df,
            rows,
            cols,
        )
        st.session_state.matrix_shape = target_shape
        st.session_state.result_payload = None


def format_component(value):
    array = np.array(value)
    array = np.real_if_close(array, tol=1000)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    row_labels = [f"R{index + 1}" for index in range(array.shape[0])]
    col_labels = [f"C{index + 1}" for index in range(array.shape[1])]

    if np.iscomplexobj(array):
        formatter = np.vectorize(
            lambda item: f"{item.real:.4f}{item.imag:+.4f}j"
            if abs(item.imag) > 1e-9
            else f"{item.real:.4f}"
        )
        display = pd.DataFrame(formatter(array), index=row_labels, columns=col_labels)
    else:
        display = pd.DataFrame(
            np.round(array.astype(float), 4),
            index=row_labels,
            columns=col_labels,
        )

    return display


def render_component(label, value):
    st.markdown(
        f"""
        <div class="section-label">
            <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    table_col, heatmap_col = st.columns([0.9, 1.1], gap="large")
    with table_col:
        st.dataframe(format_component(value), use_container_width=True)
    with heatmap_col:
        render_heatmap(label, value)


def get_heatmap_style(plot_values, label):
    min_value = float(np.min(plot_values))
    max_value = float(np.max(plot_values))
    is_signed = min_value < 0 < max_value
    is_binary = np.all(np.isin(np.unique(np.round(plot_values, 10)), [0.0, 1.0]))

    if is_binary or label == "P":
        cmap = sns.blend_palette(
            ["#eff6ff", "#93c5fd", "#2563eb", "#0f172a"],
            as_cmap=True,
        )
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    elif is_signed:
        limit = max(abs(min_value), abs(max_value))
        if np.isclose(limit, 0.0):
            limit = 1.0
        cmap = sns.diverging_palette(
            12,
            220,
            s=95,
            l=38,
            center="light",
            as_cmap=True,
        )
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    else:
        if np.isclose(min_value, max_value):
            min_value -= 1.0
            max_value += 1.0
        cmap = sns.blend_palette(
            ["#f8fafc", "#bae6fd", "#2dd4bf", "#0f766e", "#0f172a"],
            as_cmap=True,
        )
        norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    return cmap, norm


def style_heatmap_annotations(heatmap, plot_values, cmap, norm):
    for text, value in zip(heatmap.texts, plot_values.flatten(order="C")):
        rgba = cmap(norm(value))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        text_color = "#f8fafc" if luminance < 0.48 else "#0f172a"
        stroke_color = "#0f172a" if text_color == "#f8fafc" else "#ffffff"
        text.set_color(text_color)
        text.set_fontweight("bold")
        text.set_fontsize(9)
        text.set_path_effects(
            [pe.withStroke(linewidth=1.35, foreground=stroke_color, alpha=0.55)]
        )


def render_heatmap(label, value):
    array = np.array(value)
    array = np.real_if_close(array, tol=1000)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if np.iscomplexobj(array):
        st.caption(
            f"{label} heatmap uses the real component because heatmaps require real values."
        )
        plot_values = array.real.astype(float)
    else:
        plot_values = array.astype(float)

    cmap, norm = get_heatmap_style(plot_values, label)
    figure_width = max(5.4, array.shape[1] * 1.14)
    figure_height = max(3.6, array.shape[0] * 0.86)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("#f8fbff")
    ax.set_facecolor("#f8fbff")

    heatmap = sns.heatmap(
        plot_values,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        linewidths=1.2,
        linecolor="#f8fafc",
        cbar=True,
        square=False,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        cbar_kws={"shrink": 0.82, "pad": 0.02},
        ax=ax,
    )

    style_heatmap_annotations(heatmap, plot_values, cmap, norm)

    ax.set_title(
        f"{label} Heatmap",
        color="#111827",
        fontsize=13,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("Columns", color="#475569", fontsize=10, labelpad=10)
    ax.set_ylabel("Rows", color="#475569", fontsize=10, labelpad=10)
    ax.set_xticklabels(
        [f"C{index + 1}" for index in range(array.shape[1])],
        rotation=0,
        fontsize=9,
        color="#334155",
    )
    ax.set_yticklabels(
        [f"R{index + 1}" for index in range(array.shape[0])],
        rotation=0,
        fontsize=9,
        color="#334155",
    )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = heatmap.collections[0].colorbar
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, colors="#334155", length=0)
    colorbar.ax.set_facecolor("#f8fbff")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_chip(text, variant="neutral"):
    st.markdown(
        f'<span class="chip chip-{variant}">{text}</span>',
        unsafe_allow_html=True,
    )


def render_summary_cards(summary):
    cards = [
        ("Shape", summary["shape"]),
        ("Rank", str(summary["rank"])),
        ("Square", "Yes" if summary["is_square"] else "No"),
        (
            "Determinant",
            f"{summary['determinant']:.4f}"
            if summary["determinant"] is not None
            else "N/A",
        ),
    ]

    columns = st.columns(len(cards), gap="small")
    for column, (label, value) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_method_requirements(method_name):
    st.markdown("#### Method Requirements")
    for requirement in METHOD_REQUIREMENTS[method_name]:
        render_chip(requirement, "info")


def build_reconstruction(method_name, components):
    component_map = {label: np.array(value) for label, value in components}

    if method_name == "Cholesky":
        return component_map["L"] @ component_map["L^T"]
    if method_name == "LU":
        return component_map["P"] @ component_map["L"] @ component_map["U"]
    if method_name == "QR":
        return component_map["Q"] @ component_map["R"]
    if method_name == "SVD":
        return component_map["U"] @ component_map["Sigma"] @ component_map["V^T"]
    if method_name == "Eigenvalue":
        eigenvectors = component_map["Eigenvectors"]
        eigenvalues = component_map["Eigenvalues"]
        diagonal = np.diag(eigenvalues)
        return eigenvectors @ diagonal @ np.linalg.pinv(eigenvectors)

    return None


def render_validation_panel(validation):
    st.markdown("#### Validation")

    if validation["is_valid"]:
        st.success("Matrix satisfies the selected decomposition requirements.")
    else:
        st.error("Matrix does not satisfy the selected decomposition requirements yet.")

    for warning in validation["warnings"]:
        st.warning(warning)

    for error in validation["errors"]:
        st.error(error)

    summary = validation["summary"]
    status_pairs = [
        ("Finite values", summary["has_finite_values"]),
        ("Square matrix", summary["is_square"]),
        (
            "Symmetric matrix",
            summary["is_symmetric"] if summary["is_square"] else None,
        ),
        (
            "Positive definite",
            summary["is_positive_definite"] if summary["is_square"] else None,
        ),
    ]

    for label, state in status_pairs:
        if state is True:
            render_chip(f"{label}: Yes", "success")
        elif state is False:
            render_chip(f"{label}: No", "danger")
        else:
            render_chip(f"{label}: N/A", "neutral")


init_session_state()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: "Space Grotesk", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.22), transparent 32%),
            radial-gradient(circle at 88% 12%, rgba(34, 197, 94, 0.16), transparent 30%),
            linear-gradient(180deg, #f6fbff 0%, #f0f7f3 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .hero-shell {
        background:
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 32%),
            radial-gradient(circle at bottom left, rgba(34, 197, 94, 0.14), transparent 30%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 249, 0.96));
        border-radius: 28px;
        padding: 1.8rem;
        color: #0f172a;
        box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -10% -40% auto;
        width: 260px;
        height: 260px;
        background: rgba(125, 211, 252, 0.18);
        border-radius: 50%;
        filter: blur(8px);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.03em;
        color: #000000;
    }

    .hero-copy {
        color: #1e293b;
        margin: 0;
        max-width: 700px;
        line-height: 1.55;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    label, .stNumberInput label, .stSelectbox label {
        color: #000000 !important;
    }

    .glass-panel {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1.1rem 1.2rem 1.2rem 1.2rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
    }

    .metric-card {
        background: rgba(248, 250, 252, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        min-height: 88px;
    }

    .metric-label {
        color: #475569;
        font-size: 0.88rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .section-label {
        margin: 0.9rem 0 0.4rem 0;
        color: #000000;
        font-weight: 600;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.72rem;
        border-radius: 999px;
        margin: 0.15rem 0.35rem 0.15rem 0;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .chip-neutral {
        background: rgba(226, 232, 240, 0.8);
        color: #334155;
    }

    .chip-info {
        background: rgba(224, 242, 254, 0.95);
        color: #075985;
    }

    .chip-success {
        background: rgba(220, 252, 231, 0.96);
        color: #166534;
    }

    .chip-danger {
        background: rgba(254, 226, 226, 0.96);
        color: #b91c1c;
    }

    .soft-note {
        background: rgba(248, 250, 252, 0.92);
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        padding: 1rem;
        color: #334155;
    }

    .stButton > button {
        border-radius: 16px;
        min-height: 3rem;
        font-weight: 700;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        box-shadow: 0 14px 24px rgba(15, 23, 42, 0.12);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 30px rgba(15, 23, 42, 0.16);
    }

    button[data-baseweb="tab"] {
        color: #0f172a;
        font-weight: 700;
    }

    button[data-baseweb="tab"]:nth-child(1),
    button[data-baseweb="tab"]:nth-child(2) {
        color: #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-title">Matrix Decomposition Studio</div>
        <p class="hero-copy">
            Build a matrix interactively, validate it against decomposition rules in real time,
            and inspect clean factorization results with reconstruction checks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.02, 1.38], gap="large")

with left_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Matrix Builder")
    control_cols = st.columns(2, gap="small")
    with control_cols[0]:
        row_count = int(
            st.number_input("Rows", min_value=1, max_value=10, value=3, step=1)
        )
    with control_cols[1]:
        col_count = int(
            st.number_input("Columns", min_value=1, max_value=10, value=3, step=1)
        )

    sync_matrix_shape(row_count, col_count)

    decomposition_name = st.selectbox(
        "Decomposition method",
        list(DECOMPOSITION_OPTIONS.keys()),
        help="Choose the factorization you want to apply to the matrix.",
    )
    st.caption(DECOMPOSITION_OPTIONS[decomposition_name]["subtitle"])

    render_method_requirements(decomposition_name)

    st.markdown("#### Random Fill")
    random_cols = st.columns(3, gap="small")
    with random_cols[0]:
        random_min = int(
            st.number_input(
                "Min value",
                min_value=-100,
                max_value=100,
                value=-5,
                step=1,
            )
        )
    with random_cols[1]:
        random_max = int(
            st.number_input(
                "Max value",
                min_value=-100,
                max_value=100,
                value=9,
                step=1,
            )
        )
    with random_cols[2]:
        integers_only = st.toggle("Integers only", value=True)

    random_fill_disabled = random_min > random_max
    if random_fill_disabled:
        st.warning("Min value must be less than or equal to max value.")

    random_fill_clicked = st.button(
        "Generate Random Matrix",
        use_container_width=True,
        disabled=random_fill_disabled,
    )
    if random_fill_clicked:
        st.session_state.matrix_df = generate_random_dataframe(
            row_count,
            col_count,
            random_min,
            random_max,
            integers_only,
        )
        st.session_state.result_payload = None

    edited_df = st.data_editor(
        st.session_state.matrix_df,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        column_config={
            column: st.column_config.NumberColumn(
                column,
                format="%.4f",
                step=0.5,
            )
            for column in st.session_state.matrix_df.columns
        },
    )
    st.session_state.matrix_df = edited_df

    decompose_clicked = st.button(
        "Run Decomposition",
        type="primary",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

try:
    current_matrix = dataframe_to_matrix(st.session_state.matrix_df)
    current_summary = summarize_matrix(current_matrix)
    validation = validate_for_method(current_matrix, decomposition_name)
except ValueError as exc:
    current_matrix = None
    current_summary = None
    validation = {
        "summary": None,
        "requirements": METHOD_REQUIREMENTS[decomposition_name],
        "errors": [str(exc)],
        "warnings": [],
        "is_valid": False,
    }

if decompose_clicked:
    if not validation["is_valid"]:
        st.session_state.result_payload = {
            "kind": "error",
            "message": "Please fix the matrix validation errors before decomposing.",
        }
    else:
        try:
            result = DECOMPOSITION_OPTIONS[decomposition_name]["fn"](current_matrix)
            reconstruction = build_reconstruction(
                decomposition_name,
                result["components"],
            )
            reconstruction_error = None
            if reconstruction is not None:
                difference = np.array(current_matrix) - np.array(reconstruction)
                reconstruction_error = float(np.max(np.abs(difference)))

            st.session_state.result_payload = {
                "kind": "success",
                "method_name": decomposition_name,
                "result": result,
                "matrix": current_matrix,
                "reconstruction": reconstruction,
                "reconstruction_error": reconstruction_error,
            }
        except ValueError as exc:
            st.session_state.result_payload = {
                "kind": "error",
                "message": str(exc),
            }
        except Exception as exc:
            st.session_state.result_payload = {
                "kind": "error",
                "message": f"Unable to decompose the matrix: {exc}",
            }

with right_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Validation and Results")

    if current_summary is not None:
        render_summary_cards(current_summary)
        render_validation_panel(validation)
    else:
        st.error(validation["errors"][0])

    payload = st.session_state.result_payload
    if not payload:
        st.markdown(
            """
            <div class="soft-note">
                Fill the matrix, choose a method, and run the decomposition to see labeled
                factors, matrix diagnostics, and a reconstruction quality check here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif payload["kind"] == "error":
        st.error(payload["message"])
    else:
        result = payload["result"]
        st.success(result["message"])

        if payload["reconstruction_error"] is not None:
            st.info(
                "Maximum reconstruction difference: "
                f"{payload['reconstruction_error']:.8f}"
            )

        tabs = st.tabs(["Input Matrix", "Decomposed Factors", "Reconstruction"])

        with tabs[0]:
            render_component("A", payload["matrix"])

        with tabs[1]:
            for label, value in result["components"]:
                render_component(label, value)

        with tabs[2]:
            if payload["reconstruction"] is None:
                st.caption("Reconstruction is not available for this decomposition.")
            else:
                render_component("A (reconstructed)", payload["reconstruction"])

    st.markdown("</div>", unsafe_allow_html=True)
