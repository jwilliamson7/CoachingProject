import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class SVDImputer(BaseEstimator, TransformerMixin):
    """Leakage-free iterative truncated-SVD imputer with fit/transform separation.

    The low-rank factorization and the mean-initialization values are learned
    from the training partition only (``fit``), then applied unchanged to any
    matrix (``transform``). This removes two leaks present in the original
    full-matrix imputation:

      1. Train/test leakage -- the SVD basis previously saw held-out rows
         because imputation was run once on the whole 656-row matrix before
         splitting.
      2. Target leakage -- the original routine factorized every numeric
         column, which included ``Coach Tenure Class`` (the prediction target)
         and ``Avg 2Y Win Pct``. This imputer should be fit on feature columns
         only, so outcome information never enters the reconstruction.

    Parameters mirror the original ``matrix_factorization_imputation`` routine
    (50 components, mean initialization, iterative SVD to convergence) so that a
    full-sample fit reproduces the previous behaviour up to the removal of the
    leaked columns.
    """

    def __init__(self, n_components=50, max_iter=50, tol=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        rank = min(self.n_components, n_samples, n_features - 1)
        self.n_components_ = rank

        missing = np.isnan(X)
        # Column means from observed (training) entries; 0 for all-missing cols
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        self.col_means_ = col_means

        current = X.copy()
        if missing.any():
            current[missing] = np.take(col_means, np.where(missing)[1])

        prev = None
        svd = None
        for _ in range(self.max_iter):
            svd = TruncatedSVD(n_components=rank, random_state=self.random_state)
            reduced = svd.fit_transform(current)
            recon = svd.inverse_transform(reduced)
            new = current.copy()
            if missing.any():
                new[missing] = recon[missing]
            if prev is not None and missing.any():
                diff = np.mean(np.abs(new[missing] - prev[missing]))
                if diff < self.tol:
                    current = new
                    break
            prev = new.copy()
            current = new
            if not missing.any():
                break

        self.components_ = svd.components_  # (rank, n_features), learned on train
        self.explained_variance_ratio_sum_ = float(svd.explained_variance_ratio_.sum())
        self.train_imputed_ = current
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        missing = np.isnan(X)
        current = X.copy()
        if not missing.any():
            return current
        current[missing] = np.take(self.col_means_, np.where(missing)[1])

        comp = self.components_  # fixed training basis (no refit on new data)
        prev = None
        for _ in range(self.max_iter):
            recon = (current @ comp.T) @ comp
            new = current.copy()
            new[missing] = recon[missing]
            if prev is not None:
                diff = np.mean(np.abs(new[missing] - prev[missing]))
                if diff < self.tol:
                    current = new
                    break
            prev = new.copy()
            current = new
        return current

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def matrix_factorization_imputation(data, n_components=50, max_iter=200, random_state=42):
    """
    Perform matrix factorization-based imputation using Singular Value Decomposition (SVD).
    
    This method is appropriate for normalized data (including negative values) such as 
    z-score normalized sports statistics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data with missing values
    n_components : int
        Number of components for matrix factorization
    max_iter : int
        Maximum number of iterations for iterative imputation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Imputed data
    """
    print(f"Starting matrix factorization imputation...")
    print(f"Data shape: {data.shape}")
    print(f"Total missing values: {data.isnull().sum().sum()}")
    
    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Non-numeric columns: {len(non_numeric_cols)}")
    
    # Store non-numeric data separately
    non_numeric_data = data[non_numeric_cols].copy()
    numeric_data = data[numeric_cols].copy()
    
    # Check if we have enough data for matrix factorization
    missing_ratio = numeric_data.isnull().sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
    print(f"Missing data ratio: {missing_ratio:.3f}")
    
    if missing_ratio > 0.8:
        print("Warning: High missing data ratio. Results may be unreliable.")
    
    # Iterative imputation using SVD for better handling of missing data
    # Start with initial mean imputation
    current_data = numeric_data.copy()
    initial_imputer = SimpleImputer(strategy='mean')
    current_data = pd.DataFrame(
        initial_imputer.fit_transform(current_data),
        columns=numeric_data.columns,
        index=numeric_data.index
    )
    
    # Determine optimal number of components (shouldn't exceed min dimension)
    n_components = min(n_components, current_data.shape[0], current_data.shape[1] - 1)
    print(f"Using {n_components} components for matrix factorization")
    
    # Iterative imputation with SVD
    missing_mask = numeric_data.isnull()
    prev_imputed = None
    
    print("Starting iterative SVD imputation...")
    for iteration in range(min(max_iter, 50)):  # Limit iterations for stability
        try:
            # Apply SVD to current data
            svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            U_reduced = svd.fit_transform(current_data)
            reconstructed = svd.inverse_transform(U_reduced)
            
            # Create reconstructed DataFrame
            reconstructed_df = pd.DataFrame(
                reconstructed,
                columns=numeric_data.columns,
                index=numeric_data.index
            )
            
            # Only update missing values with reconstructed values
            new_imputed = current_data.copy()
            new_imputed[missing_mask] = reconstructed_df[missing_mask]
            
            # Check for convergence
            if prev_imputed is not None:
                diff = np.mean(np.abs(new_imputed.values[missing_mask] - prev_imputed.values[missing_mask]))
                print(f"Iteration {iteration + 1}: Mean absolute difference = {diff:.6f}")
                
                if diff < 1e-6:  # Convergence threshold
                    print(f"Converged after {iteration + 1} iterations")
                    break
            else:
                print(f"Iteration {iteration + 1}: Initial reconstruction")
            
            prev_imputed = new_imputed.copy()
            current_data = new_imputed
            
        except Exception as e:
            print(f"SVD iteration {iteration + 1} failed: {e}")
            if iteration == 0:
                print("Falling back to mean imputation...")
                current_data = pd.DataFrame(
                    initial_imputer.fit_transform(numeric_data),
                    columns=numeric_data.columns,
                    index=numeric_data.index
                )
            break
    
    reconstructed_df = current_data
    print(f"SVD imputation completed with explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Create final imputed dataset
    # Only replace missing values, keep original non-missing values
    final_numeric_data = numeric_data.copy()
    missing_mask = numeric_data.isnull()
    final_numeric_data[missing_mask] = reconstructed_df[missing_mask]
    
    # Combine numeric and non-numeric data
    if len(non_numeric_cols) > 0:
        result = pd.concat([non_numeric_data, final_numeric_data], axis=1)
        # Restore original column order
        result = result[data.columns]
    else:
        result = final_numeric_data
    
    print(f"Imputation completed. Remaining missing values: {result.isnull().sum().sum()}")
    
    return result

def main():
    """Main function to run matrix factorization imputation on master_data.csv"""
    print("SVD-Based Matrix Factorization Imputation for NFL Coaching Data")
    print("=" * 65)

    # Define paths relative to project root
    data_dir = project_root / "data"
    input_file = data_dir / "master_data.csv"

    # Load the data
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data from {input_file}: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Store the unnamed index column if present
    unnamed_index_col = None
    if 'Unnamed: 0' in df.columns:
        unnamed_index_col = df['Unnamed: 0'].copy()
        df = df.drop('Unnamed: 0', axis=1)
    
    # Perform matrix factorization imputation
    try:
        imputed_df = matrix_factorization_imputation(
            df, 
            n_components=50,  # Reduced for stability
            max_iter=300,
            random_state=42
        )
        
        # Add back the unnamed index column if it existed
        if unnamed_index_col is not None:
            imputed_df.insert(0, 'Unnamed: 0', unnamed_index_col)
        
        # Save the imputed data
        output_file = data_dir / 'svd_imputed_master_data.csv'
        imputed_df.to_csv(output_file, index=False)
        print(f"\nImputed data saved to: {output_file}")
        
        # Print summary statistics
        print("\nImputation Summary:")
        print("-" * 30)
        print(f"Original missing values: {df.isnull().sum().sum()}")
        print(f"Remaining missing values: {imputed_df.isnull().sum().sum()}")
        print(f"Missing values reduced by: {df.isnull().sum().sum() - imputed_df.isnull().sum().sum()}")
        
        # Show columns with most missing values before and after
        original_missing = df.isnull().sum().sort_values(ascending=False)
        imputed_missing = imputed_df.isnull().sum().sort_values(ascending=False)
        
        print(f"\nTop 10 columns by missing values (before -> after):")
        for col in original_missing.head(10).index:
            print(f"{col}: {original_missing[col]} -> {imputed_missing[col]}")
            
    except Exception as e:
        print(f"Error during imputation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()