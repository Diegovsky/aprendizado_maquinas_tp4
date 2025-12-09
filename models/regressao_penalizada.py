"""Module: Penalized Regression (Lasso, Ridge, ElasticNet)"""
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from .shared_utils import load_and_process_data, split_data, print_split_info, save_model, load_model


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], StandardScaler]:
    """Data preparation pipeline: Split + StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, X_test, y_train, y_test), scaler


def train_penalized_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    penalty: str = 'l2',
    solver: str = 'lbfgs',
    l1_ratio: float = None,
    cv_splits: int = 5,
) -> LogisticRegressionCV:
    """Trains a penalized logistic regression model with safe defaults."""
    model = LogisticRegressionCV(
        Cs=[0.1, 1.0, 10.0],
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        max_iter=1000,
        cv=cv_splits,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


def extract_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
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


def visualize_comparison(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    save_path: str = 'results_comparison.png'
) -> None:
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


def run(file_path: str = None, target_col: str = 'target') -> Dict[str, Any]:
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

    # Determine a safe number of CV splits based on smallest class count
    class_counts = np.unique(y_train, return_counts=True)[1]
    min_class = class_counts.min() if class_counts.size else 0
    cv_splits = max(2, min(5, int(min_class))) if min_class > 1 else 2

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
        model_key = f'penalized_regression_{name.replace(" ", "_")}'
        try:
            model = load_model(model_key)
            if model is None:
                try:
                    model = train_penalized_model(X_train, y_train, penalty, solver, l1_ratio, cv_splits=cv_splits)
                except Exception:
                    # Fallback to simple logistic regression without CV
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(
                        penalty='l2',
                        solver='lbfgs',
                        max_iter=500,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                save_model(model, model_key)
            metrics = extract_metrics(model, X_test, y_test)
            results_models[name] = model
            model_metrics[name] = metrics
        except Exception:
            continue

    # Generate comparison plot
    if results_models:
        visualize_comparison(results_models, X_test, y_test, feats)
    
    if not model_metrics:
        return {'error': 'No penalized regression models could be trained.'}

    # Choose best model by (auc, accuracy)
    best_key = max(model_metrics.items(), key=lambda item: (item[1].get('auc', 0.0), item[1].get('accuracy', 0.0)))[0]
    best_model = results_models[best_key]

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return {
        'accuracy': model_metrics[best_key]['accuracy'],
        'auc': model_metrics[best_key]['auc'],
        'report': report
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
