"""
Module: Decision Tree with Cost-Complexity Pruning
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from .shared_utils import load_and_process_data, split_data, print_split_info


def train_decision_tree(X_train, y_train):
    """Train a Decision Tree with Cost-Complexity Pruning."""
    clf_base = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    path = clf_base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    ccp_alphas = ccp_alphas[::5] if len(ccp_alphas) > 50 else ccp_alphas
    
    grid_search = GridSearchCV(
        estimator=clf_base,
        param_grid={'ccp_alpha': ccp_alphas},
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    except Exception as e:
        clf_base.fit(X_train, y_train)
        return clf_base


def visualize_tree_results(model, X_test, y_test, feature_names, save_prefix='tree_results'):
    """Generate visualization plots for decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=[str(c) for c in np.unique(y_test)], 
              filled=True, 
              rounded=True, 
              fontsize=10,
              max_depth=4)
    
    plt.savefig(f"{save_prefix}_structure.png")
    plt.close()
    
    # Feature importances
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title("Feature Importance")
    plt.xlabel("Relative Importance")
    
    plt.savefig(f"{save_prefix}_features.png")
    plt.close()


def run(file_path=None):
    """Run decision tree pipeline."""
    X, y, features = load_and_process_data(file_path)
    
    if len(y) < 50:
        raise ValueError("Dataset too small (< 50 samples)")

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

    model = train_decision_tree(X_train, y_train)

    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.5

    acc = accuracy_score(y_test, y_pred)
    depth = model.get_depth()
    n_leaves = model.get_n_leaves()

    results = {
        'accuracy': acc,
        'auc': auc,
        'depth': depth,
        'leaves': n_leaves,
        'report': classification_report(y_test, y_pred)
    }
    
    visualize_tree_results(model, X_test, y_test, features)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree Training")
    parser.add_argument('--arquivo', type=str, default=None, help="Path to CSV or TSV file")
    args = parser.parse_args()

    results = run(file_path=args.arquivo)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
