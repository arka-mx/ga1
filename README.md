# Matrix Decomposition Studio

A Streamlit app that lets you enter any matrix, choose a decomposition, and
watch every single calculation unfold step-by-step — exactly like Python Tutor
shows program execution, but for linear algebra.

## Features

- **Six decompositions:** LU, QR, Eigenvalue, SVD, Cholesky, PCA
- **Step-by-step visualiser:** slider through every arithmetic step with the
  exact formula used at each stage and an animated heatmap showing which cell
  was just computed
- **Matrix properties panel:** shape, rank, determinant, symmetry, positive-definiteness
- **Reconstruction check:** verifies A = factors (within floating-point tolerance)
- **PCA extras:** explained-variance bar chart per principal component
- **Download:** export any output matrix as CSV
- **Three input modes:** manual entry, CSV upload, random generation

## Project structure

```
matrix_studio/
├── app.py                   # main Streamlit application
├── requirements.txt
├── decomposition/
│   ├── __init__.py
│   ├── lu.py                # Gaussian elimination + step log
│   ├── qr.py                # Gram-Schmidt + step log
│   ├── svd.py               # eigen-based SVD + step log
│   ├── eigen.py             # QR iteration + step log
│   ├── cholesky.py          # Banachiewicz algorithm + step log
│   └── pca.py               # full PCA from scratch + step log
└── utils/
    ├── __init__.py
    ├── validators.py        # input validation, matrix property checks
    └── helpers.py           # heatmap helpers, random matrix generation
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes on the merged codebase

- **A1 (Assignment1)** had better manual implementations of each algorithm
  (Gram-Schmidt QR, Banachiewicz Cholesky, QR-iteration eigenvalues) and
  richer UI with heatmaps and downloads.
- **GA1** had cleaner object-oriented structure, better error messages, and
  a nicer CSS-styled layout.
- This merged version takes the manual algorithms from A1 (educational value),
  the validation and property-checking logic from GA1, and adds the
  step-by-step visualiser and PCA on top.
