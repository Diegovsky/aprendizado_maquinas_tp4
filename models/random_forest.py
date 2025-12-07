"""
Module: Random Forest with Hyperparameter Optimization
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay

from .shared_utils import load_and_process_data, split_data, print_split_info, save_model, load_model


def train_random_forest_optimized(X_train, y_train):
    """Train a robust Random Forest with hyperparameter optimization."""
    rf = RandomForestClassifier(random_state=42, oob_score=True, class_weight='balanced')
    
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    
    try:
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_
    except Exception as e:
        rf.fit(X_train, y_train)
        return rf


def visualize_forest_performance(model, X_test, y_test, feature_names, save_prefix='rf_results'):
    """Generate importance and confusion matrix visualizations."""
    plt.figure(figsize=(18, 12))
    
    # MDI Importance
    plt.subplot(2, 2, 1)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Feature Importance (MDI)')
    plt.xlabel('Relative Importance')
    
    # Permutation Importance
    plt.subplot(2, 2, 2)
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[-15:]
    
    plt.boxplot(result.importances[sorted_idx].T, vert=False,
                labels=[feature_names[i] for i in sorted_idx])
    plt.title('Permutation Importance')
    plt.xlabel('Accuracy Drop')

    # Confusion Matrix
    ax_cm = plt.subplot(2, 1, 2)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        display_labels=['Class 0', 'Class 1'],
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax_cm
    )

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_results.png')
    plt.close()


def run(file_path=None):
    """Run Random Forest pipeline."""
    X, y, features = load_and_process_data(file_path, n_features=25, n_informative=15)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

    model = load_model('random_forest')
    if model is None:
        model = train_random_forest_optimized(X_train, y_train)
        save_model(model, 'random_forest')

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
    
    visualize_forest_performance(model, X_test, y_test, features)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    args = parser.parse_args()
    
    results = run(file_path=args.arquivo)
    print(f'Accuracy: {results["accuracy"]:.4f}')
    print(f'AUC: {results["auc"]:.4f}')
