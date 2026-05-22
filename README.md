# Matrix Decomposition Studio

Matrix Decomposition Studio is a Streamlit app for building matrices interactively, validating whether a selected factorization can be applied, and visualizing the resulting components as tables and heatmaps.

The project currently supports:

- Cholesky decomposition
- Eigenvalue decomposition
- LU decomposition
- QR decomposition
- Singular Value Decomposition (SVD)

## Features

- Interactive matrix editor with configurable row and column counts
- Random matrix generation with integer or decimal values
- Real-time validation based on the selected decomposition
- Matrix summary cards for shape, rank, square status, and determinant
- Factor visualization using tables and heatmaps
- Reconstruction check to compare the original matrix with the decomposed result

## Setup Instructions

### 1. Clone or open the project

Make sure you are inside the project folder:

```powershell
cd "E:\Algo and Python\GA1"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Command Prompt:

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```powershell
streamlit run app.py
```

After launching, Streamlit will open the app in your browser or provide a local URL such as `http://localhost:8501`.

## Project Structure

```text
GA1/
|-- app.py
|-- README.md
|-- requirements.txt
|-- decomposition/
|   |-- cholesky.py
|   |-- eigen.py
|   |-- lu.py
|   |-- methods.py
|   |-- qr.py
|   |-- svd.py
|-- utils/
|   |-- helpers.py
|   |-- requirements.txt
|   |-- valiadators.py
```

### Folder and File Roles

- `app.py`: Main Streamlit interface for matrix input, validation, decomposition, reconstruction, and plotting.
- `decomposition/`: Contains one implementation file per matrix decomposition method.
- `decomposition/methods.py`: Creates a single access point for all decomposition classes.
- `utils/valiadators.py`: Converts table input to numeric matrices, summarizes matrices, and validates decomposition rules.
- `utils/helpers.py`: Present in the project structure but currently not used by the main app.
- `requirements.txt`: Main Python dependency list for the Streamlit app.

## Screenshots

No screenshot image files are currently committed in the repository.

If you want this section to include actual images, the most useful screenshots for this project would be:

1. Matrix Builder panel with a custom input matrix
2. Validation panel showing method requirements and error messages
3. Successful decomposition output with factor tables and heatmaps
4. Reconstruction tab showing the rebuilt matrix

Example markdown once screenshots are added:

```md
![Matrix Builder](screenshots/matrix-builder.png)
![LU Decomposition Result](screenshots/lu-result.png)
```

## Explanation of Decomposition Methods

### 1. Cholesky Decomposition

Cholesky decomposition factors a matrix as:

```text
A = L L^T
```

Where:

- `A` is a square matrix
- `L` is a lower triangular matrix
- `L^T` is the transpose of `L`

In this project, Cholesky is only allowed when the matrix is:

- Square
- Symmetric
- Positive definite

This is useful for symmetric positive definite systems and efficient numerical computation.

### 2. Eigenvalue Decomposition

Eigenvalue decomposition expresses a square matrix in terms of its eigenvalues and eigenvectors:

```text
A = V Lambda V^-1
```

Where:

- `V` contains eigenvectors
- `Lambda` contains eigenvalues

In this app, the result is shown as:

- `Eigenvalues`
- `Eigenvectors`

This method is available for square matrices.

### 3. LU Decomposition

LU decomposition breaks a matrix into permutation, lower, and upper triangular matrices:

```text
A = P L U
```

Where:

- `P` is a permutation matrix
- `L` is lower triangular
- `U` is upper triangular

In this project, LU is applied to square matrices and is useful for solving linear systems and understanding row operations.

### 4. QR Decomposition

QR decomposition factors a matrix as:

```text
A = Q R
```

Where:

- `Q` is an orthogonal matrix
- `R` is an upper triangular matrix

This project allows QR decomposition for any numeric matrix, including rectangular ones.

QR decomposition is commonly used in least-squares problems and numerical linear algebra.

### 5. Singular Value Decomposition (SVD)

SVD factors a matrix as:

```text
A = U Sigma V^T
```

Where:

- `U` contains left singular vectors
- `Sigma` contains singular values
- `V^T` contains right singular vectors

In this project, SVD works for any numeric matrix and is especially useful for:

- Dimensionality reduction
- Matrix approximation
- Noise filtering
- Data analysis

## How the App Works

1. The user selects the matrix size.
2. The matrix is entered manually or generated randomly.
3. The selected decomposition method determines the validation rules.
4. The app checks whether the matrix satisfies those rules.
5. If valid, the decomposition is computed and displayed.
6. The original and reconstructed matrices are compared to show the reconstruction error.

## Dependencies

The project uses the following libraries:

- `streamlit`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`

## Notes

- Matrix input is converted to numeric values before decomposition.
- Non-finite or invalid values are rejected during validation.
- Heatmaps use the real part of the data when a result contains complex values.
