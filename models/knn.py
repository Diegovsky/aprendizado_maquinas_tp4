"""
Module: k-Nearest Neighbors with Hyperparameter Optimization
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from .shared_utils import load_and_process_data, split_data, print_split_info, save_model, load_model


def train_knn_optimized(X_train, y_train):
    """Train a robust kNN with RandomizedSearchCV and Pipeline."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ])
    
    param_dist = {
        'knn__n_neighbors': list(range(3, 30, 2)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__p': [1, 2]
    }
    
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring='roc_auc',
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    
    try:
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_
    except Exception as e:
        pipe.fit(X_train, y_train)
        return pipe


def visualize_knn_performance(model, X_test, y_test, feature_names, save_prefix='knn_results'):
    """Generate performance visualization plots."""
    plt.figure(figsize=(18, 8))
    
    plt.subplot(1, 2, 1)
    result_perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx_perm = result_perm.importances_mean.argsort()[-15:]
    
    plt.boxplot(
        result_perm.importances[sorted_idx_perm].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx_perm]
    )
    plt.title("Feature Importance (Permutation)")
    plt.xlabel("Accuracy Drop")

    ax_cm = plt.subplot(1, 2, 2)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        display_labels=["Class 0", "Class 1"],
        cmap=plt.cm.Greens,
        normalize='true',
        ax=ax_cm
    )
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_results.png")
    plt.close()


def run(file_path=None):
    """Run kNN pipeline."""
    X, y, features = load_and_process_data(file_path, n_features=20)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, stratify=True)

    model = load_model('knn')
    if model is None:
        model = train_knn_optimized(X_train, y_train)
        save_model(model, 'knn')

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.5
    
    results = {
        'accuracy': acc,
        'auc': auc,
        'report': classification_report(y_test, y_pred)
    }
    
    visualize_knn_performance(model, X_test, y_test, features)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    args = parser.parse_args()
    
    results = run(file_path=args.arquivo)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
