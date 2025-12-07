"""
Module: Histogram Gradient Boosting Classifier
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from .shared_utils import load_and_process_data, split_data, save_model, load_model

def train_hgbm_optimized(X_train, y_train):
    hgbm = HistGradientBoostingClassifier(random_state=42, early_stopping='auto')
    param_dist = {
        'max_iter': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_regularization': [0.0, 0.1, 0.5]
    }
    random_search = RandomizedSearchCV(
        estimator=hgbm,
        param_distributions=param_dist,
        n_iter=8,
        scoring='roc_auc',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    try:
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_
    except Exception:
        hgbm.fit(X_train, y_train)
        return hgbm

def visualize_boosting_performance(model, X_test, y_test, feature_names, save_prefix='gbm_results'):
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[-15:]
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], color='lightcoral')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Permutation Importance')
    plt.xlabel('Importance')
    plt.subplot(2, 1, 2)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['Class 0', 'Class 1'],
        cmap=plt.cm.Blues,
        normalize='true'
    )
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_results.png')
    plt.close()

def run(file_path=None):
    X, y, features = load_and_process_data(file_path, n_features=25, n_informative=15)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    model = load_model('gbte')
    if model is None:
        model = train_hgbm_optimized(X_train, y_train)
        save_model(model, 'gbte')
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
    visualize_boosting_performance(model, X_test, y_test, features)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    args = parser.parse_args()
    results = run(file_path=args.arquivo)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
