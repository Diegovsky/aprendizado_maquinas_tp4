"""
Módulo compartilhado com funções comuns para todos os modelos ML.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_data_from_file(file_path, target_col='target'):
    """
    Load data from TSV or CSV file and perform preprocessing.
    
    Args:
        file_path: Path to the file
        target_col: Name of the target column (default: 'target')
    
    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
        feature_names: Feature names (list)
    """
    try:
        try:
            df = pd.read_csv(file_path, sep='\t')
        except:
            df = pd.read_csv(file_path, sep=',')

        # Identify target column
        if target_col not in df.columns:
            target_col = df.columns[-1]
        
        # Remove rows with missing target values
        df = df.dropna(subset=[target_col])
        
        # One-Hot Encoding for categorical variables
        df = pd.get_dummies(df, drop_first=True, dtype=int)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Remove constant columns (zero variance)
        cols_constantes = [col for col in X.columns if X[col].nunique() <= 1]
        if cols_constantes:
            X = X.drop(columns=cols_constantes)
        
        feature_names = X.columns.tolist()
        X_values = X.values.astype(float)
        
        return X_values, y, feature_names
        
    except Exception as e:
        raise Exception(f"Critical error processing file: {e}")


def generate_synthetic_data(n_samples=1000, n_features=10, n_informative=5, random_state=42):
    """
    Generate synthetic data for testing and demos.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        random_state: Seed for reproducibility
    
    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
        feature_names: Feature names (list)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(1, n_features // 5),
        random_state=random_state
    )
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    return X, y, feature_names


def load_and_process_data(file_path=None, n_samples=1000, n_features=10, 
                          n_informative=5, target_col='target'):
    """
    Generic function that loads file or generates synthetic data.
    
    Args:
        file_path: Path to file (if None, uses synthetic data)
        n_samples: Number of synthetic samples
        n_features: Number of synthetic features
        n_informative: Number of synthetic informative features
        target_col: Name of target column
    
    Returns:
        X: Features
        y: Target
        feature_names: Feature names
    """
    if file_path:
        return load_data_from_file(file_path, target_col)
    else:
        return generate_synthetic_data(n_samples, n_features, n_informative)


def split_data(X, y, test_size=0.3, random_state=42, stratify=True):
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Seed for reproducibility
        stratify: If True, uses stratification
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_arg = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, 
                           stratify=stratify_arg)


def print_split_info(X_train, X_test, y_train, y_test):
    """
    Print information about train/test split.
    """
    pass
