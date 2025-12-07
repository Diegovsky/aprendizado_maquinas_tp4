"""Module: Penalized Regression (Lasso, Ridge, ElasticNet)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from .shared_utils import load_and_process_data, split_data, print_split_info, save_model, load_model


def prepare_data(X, y, test_size=0.3, random_state=42):
    """Data preparation pipeline: Split + StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, X_test, y_train, y_test), scaler


def train_penalized_model(X_train, y_train, penalty='l2', solver='lbfgs', l1_ratio=None):
    """Trains a penalized logistic regression model."""
    model = LogisticRegressionCV(
        Cs=[0.1, 1.0, 10.0],
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        max_iter=1000,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def extract_metrics(model, X_test, y_test):
    """Generates a simple performance report."""
    acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.0
    
    return {
        "accuracy": acc,
        "auc": auc,
        "coefficients": model.coef_.flatten()
    }


def visualize_comparison(models_dict, X_test, y_test, feature_names, save_path='results_comparison.png'):
    """Creates visual comparison of models."""
    plt.figure(figsize=(14, 8))
    
    markers = ['s-', 'o-', '^-'] 
    
    for (name, model), marker in zip(models_dict.items(), markers):
        coefs = np.abs(model.coef_.flatten())
        plt.plot(coefs[:min(len(coefs), 20)], marker, label=name, alpha=0.7, linewidth=2)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8) 
    plt.title("Model Comparison: Regularization Impact")
    plt.xlabel("Feature Index")
    plt.ylabel("Absolute Weight Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)


def run(file_path=None, target_col='target'):
    """Run penalized regression pipeline."""
    
    if file_path:
        try:
            X, y, feats = load_and_process_data(file_path, target_col=target_col)
        except Exception as e:
            return {'error': str(e)}
    else:
        X, y, feats = load_and_process_data(n_samples=5000, n_features=50, n_informative=10)
    
    # Data preparation
    (X_train, X_test, y_train, y_test), scaler = prepare_data(X, y)

    # Experiment configurations
    configs = [
        ("L2 (Ridge)", "l2", "lbfgs", None),      
        ("L1 (Lasso)", "l1", "saga", None),       
        ("ElasticNet", "elasticnet", "saga", 0.5) 
    ]
    
    results_models = {}
    model_metrics = {}
    
    # Training loop
    for name, penalty, solver, l1_ratio in configs:
        try:
            model = load_model(f'penalized_regression_{name.replace(" ", "_")}')
            if model is None:
                model = train_penalized_model(X_train, y_train, penalty, solver, l1_ratio)
                save_model(model, f'penalized_regression_{name.replace(" ", "_")}')
            metrics = extract_metrics(model, X_test, y_test)
            results_models[name] = model
            model_metrics[name] = metrics
        except Exception as e:
            pass

    # Generate comparison plot
    if results_models:
        visualize_comparison(results_models, X_test, y_test, feats)
    
    return {
        'accuracy': model_metrics[list(model_metrics.keys())[0]]['accuracy'] if model_metrics else 0.0,
        'auc': model_metrics[list(model_metrics.keys())[0]]['auc'] if model_metrics else 0.0,
        'report': f"Trained {len(results_models)} models"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Penalized Regression Pipeline")
    parser.add_argument('--arquivo', type=str, help='Path to TSV file')
    parser.add_argument('--target', type=str, default='target', help='Target column name')
    
    args = parser.parse_args()
    
    results = run(file_path=args.arquivo, target_col=args.target)
    if 'error' not in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
