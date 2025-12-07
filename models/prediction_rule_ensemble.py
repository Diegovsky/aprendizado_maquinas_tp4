"""Module: RuleFit - Prediction Rule Ensemble (Simplified Version)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, classification_report

from .shared_utils import load_and_process_data, split_data


class RuleFitSimple:
    """Simplified RuleFit implementation for educational purposes."""
    
    def __init__(self, n_rules=100, random_state=42):
        self.n_rules = n_rules
        self.random_state = random_state
        self.gb = None
        self.lasso = None
        
    def fit(self, X, y, mode='classify'):
        """Trains the RuleFit model."""
        if mode == 'classify':
            self.gb = GradientBoostingClassifier(
                n_estimators=50, random_state=self.random_state, max_depth=5
            )
        else:
            self.gb = GradientBoostingRegressor(
                n_estimators=50, random_state=self.random_state, max_depth=5
            )
        
        self.gb.fit(X, y)
        
        # Extract leaves from trees as features
        leaf_features = self.gb.apply(X)
        X_rules = leaf_features.reshape(X.shape[0], -1)
        
        if mode == 'classify':
            self.lasso = LogisticRegressionCV(penalty='l1', solver='saga', max_iter=1000, cv=3)
        else:
            self.lasso = LassoCV(cv=5, max_iter=1000)
        
        self.lasso.fit(X_rules, y)
        
    def predict(self, X):
        """Makes predictions."""
        leaf_features = self.gb.apply(X)
        X_rules = leaf_features.reshape(X.shape[0], -1)
        return self.lasso.predict(X_rules)
    
    def predict_proba(self, X):
        """Probabilistic predictions (classification only)."""
        if not hasattr(self.lasso, 'predict_proba'):
            raise AttributeError("predict_proba not available for regression")
        leaf_features = self.gb.apply(X)
        X_rules = leaf_features.reshape(X.shape[0], -1)
        return self.lasso.predict_proba(X_rules)


def train_rulefit(X_train, y_train, mode='classify'):
    """Trains a RuleFit model."""
    model = RuleFitSimple(n_rules=100, random_state=42)
    model.fit(X_train, y_train, mode=mode)
    return model


def visualize_rules(model, X_test, y_test, feature_names, save_prefix='results', mode='classify'):
    """Visualizes rules and performance."""
    
    plt.figure(figsize=(16, 10))
    
    # Feature importance
    plt.subplot(2, 2, 1)
    if hasattr(model.lasso, 'coef_'):
        coefs = np.abs(model.lasso.coef_.flatten())
        top_indices = np.argsort(coefs)[-15:]
        
        plt.barh(range(len(top_indices)), coefs[top_indices], color='steelblue')
        plt.yticks(range(len(top_indices)), [f"Rule_{i}" for i in top_indices])
        plt.title("Top 15 Rules by Coefficient")
        plt.xlabel("Coefficient Magnitude")
    
    # Coefficient distribution
    plt.subplot(2, 2, 2)
    if hasattr(model.lasso, 'coef_'):
        plt.hist(model.lasso.coef_.flatten(), bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Coefficient Magnitude")
        plt.ylabel("Frequency")
        plt.title("Coefficient Distribution")
        plt.grid(True, alpha=0.3, axis='y')
    
    # Performance metrics
    plt.subplot(2, 2, 3)
    y_pred = model.predict(X_test)
    
    if mode == 'classify':
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            scores = np.column_stack([y_pred, y_proba])
        except:
            scores = y_pred.reshape(-1, 1)
        
        plt.text(0.5, 0.5, "Classification\nRuleFit", ha='center', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "Regression\nRuleFit", ha='center', fontsize=14, fontweight='bold')
    
    plt.axis('off')
    
    # Statistics
    plt.subplot(2, 2, 4)
    stats_text = "Model: RuleFit\n"
    stats_text += f"Mode: {'Classification' if mode == 'classify' else 'Regression'}\n"
    if mode == 'classify':
        acc = accuracy_score(y_test, y_pred)
        stats_text += f"Accuracy: {acc:.4f}\n"
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            stats_text += f"AUC: {auc:.4f}"
        except:
            pass
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        stats_text += f"MSE: {mse:.4f}\n"
        stats_text += f"R²: {r2:.4f}"
    
    plt.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_complete.png", dpi=100)
    plt.close()


def run(file_path=None, mode='classify'):
    """Run RuleFit pipeline."""

    try:
        X, y, feats = load_and_process_data(file_path, n_features=10)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

        model = train_rulefit(X_train, y_train, mode=mode)

        y_pred = model.predict(X_test)
        
        if mode == 'classify':
            acc = accuracy_score(y_test, y_pred)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            visualize_rules(model, X_test, y_test, feats, mode=mode)
            
            return {
                'accuracy': acc,
                'auc': auc,
                'report': classification_report(y_test, y_pred)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            visualize_rules(model, X_test, y_test, feats, mode=mode)
            
            return {
                'mse': mse,
                'r2': r2,
                'report': f'MSE: {mse:.4f}, R²: {r2:.4f}'
            }

    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified RuleFit")
    parser.add_argument('--arquivo', type=str, default=None)
    parser.add_argument('--modo', type=str, default='classify', choices=['classify', 'regress'])
    args = parser.parse_args()

    results = run(file_path=args.arquivo, mode=args.modo)
    if 'error' not in results:
        if 'accuracy' in results:
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"AUC: {results['auc']:.4f}")
        else:
            print(f"MSE: {results['mse']:.4f}")
            print(f"R²: {results['r2']:.4f}")
