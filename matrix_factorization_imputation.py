import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def matrix_factorization_imputation(data, n_components=50, max_iter=200, random_state=42):
    """
    Perform matrix factorization-based imputation using Non-negative Matrix Factorization (NMF).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data with missing values
    n_components : int
        Number of components for matrix factorization
    max_iter : int
        Maximum number of iterations for NMF
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
    
    # Initial imputation with mean values for NMF (NMF requires non-negative values)
    initial_imputer = SimpleImputer(strategy='mean')
    numeric_imputed_initial = pd.DataFrame(
        initial_imputer.fit_transform(numeric_data),
        columns=numeric_data.columns,
        index=numeric_data.index
    )
    
    # Scale the data to ensure non-negativity for NMF
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_imputed_initial)
    
    # Shift data to ensure all values are non-negative for NMF
    min_val = scaled_data.min()
    if min_val < 0:
        scaled_data = scaled_data - min_val + 0.01
    
    # Determine optimal number of components (shouldn't exceed min dimension)
    n_components = min(n_components, scaled_data.shape[0], scaled_data.shape[1])
    print(f"Using {n_components} components for matrix factorization")
    
    # Apply NMF
    try:
        nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state, 
                  init='random', solver='mu', beta_loss='frobenius')
        W = nmf.fit_transform(scaled_data)
        H = nmf.components_
        
        # Reconstruct the matrix
        reconstructed = np.dot(W, H)
        
        # Reverse the shifting and scaling
        if min_val < 0:
            reconstructed = reconstructed + min_val - 0.01
        
        reconstructed_df = pd.DataFrame(
            scaler.inverse_transform(reconstructed),
            columns=numeric_data.columns,
            index=numeric_data.index
        )
        
        print(f"NMF reconstruction error: {nmf.reconstruction_err_:.6f}")
        
    except Exception as e:
        print(f"NMF failed: {e}")
        print("Falling back to mean imputation...")
        reconstructed_df = numeric_imputed_initial
    
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
    print("Matrix Factorization Imputation for NFL Coaching Data")
    print("=" * 55)
    
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
        output_file = 'mf_imputed_master_data.csv'
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