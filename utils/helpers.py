import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time


def plot_heatmap(matrix, title="Matrix Heatmap"):
    """Render a matrix as a colour-coded heatmap with value annotations."""
    fig = px.imshow(
        matrix,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        title=title,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_heatmap_highlighted(matrix, title, highlight_cell=None, highlight_col=None):
    """
    Heatmap with an optional highlighted cell or column (for step visualiser).
    highlight_cell: (row, col) tuple
    highlight_col: int column index
    """
    fig = plot_heatmap(matrix, title)

    if highlight_cell is not None:
        r, c = highlight_cell
        fig.add_shape(
            type="rect",
            x0=c - 0.5, x1=c + 0.5,
            y0=r - 0.5, y1=r + 0.5,
            line=dict(color="yellow", width=3),
        )
    if highlight_col is not None:
        c = highlight_col
        fig.add_shape(
            type="rect",
            x0=c - 0.5, x1=c + 0.5,
            y0=-0.5, y1=matrix.shape[0] - 0.5,
            line=dict(color="yellow", width=3),
        )
    return fig


def plot_explained_variance(explained_ratios):
    """Bar chart of explained variance per principal component."""
    n = len(explained_ratios)
    cumulative = np.cumsum(explained_ratios) * 100
    individual = explained_ratios * 100

    fig = go.Figure()
    fig.add_bar(
        x=[f"PC{i+1}" for i in range(n)],
        y=individual,
        name="Individual %",
        marker_color="#4f8ef7",
    )
    fig.add_scatter(
        x=[f"PC{i+1}" for i in range(n)],
        y=cumulative,
        name="Cumulative %",
        mode="lines+markers",
        marker_color="#f7844f",
        yaxis="y",
    )
    fig.update_layout(
        title="Explained Variance by Principal Component",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def generate_random_matrix(rows, cols, kind="random"):
    """
    Generate a matrix for quick experimentation.
    kind: 'random' | 'square' | 'spd' (symmetric positive definite)
    """
    if kind == "spd":
        n = min(rows, cols)
        A = np.random.randn(n, n) * 5
        return np.round(A @ A.T + np.eye(n) * 0.5, 2)
    elif kind == "square":
        n = min(rows, cols)
        return np.round(np.random.randn(n, n) * 10, 2)
    else:
        return np.round(np.random.randn(rows, cols) * 10, 2)


def time_execution(func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0
