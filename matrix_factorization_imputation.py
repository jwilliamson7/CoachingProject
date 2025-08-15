import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

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
    
    # Load the data
    try:
        df = pd.read_csv('master_data.csv')
        print(f"Successfully loaded data: {df.shape}")
    except FileNotFoundError:
        print("Error: master_data.csv not found in current directory")
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
        output_file = 'svd_imputed_master_data.csv'
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