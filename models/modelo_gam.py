"""Module: Generalized Additive Models (GAM) with Splines"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from pygam import LogisticGAM, s, f, l

from .shared_utils import load_and_process_data, split_data, print_split_info


def build_gam_terms(X_train, n_splines=20):
    """Builds GAM terms based on automatic detection."""
    n_features = X_train.shape[1]
    terms = []
    
    for i in range(min(n_features, 5)):
        if X_train[:, i].nunique() > 10:
            terms.append(s(i, n_splines=min(n_splines, 25)))
        else:
            terms.append(f(i))
    
    return terms


def train_gam_automatic(X_train, y_train):
    """Configures and trains a GAM with intelligent term selection."""
    terms = build_gam_terms(X_train)
    gam = LogisticGAM(*terms)
    gam.fit(X_train, y_train)
    
    return gam


def visualize_gam_interpretation(gam, scaler, feature_names, X_test, y_test, save_path='gam_results.png'):
    """Plots partial dependencies with performance analysis."""
    plt.figure(figsize=(18, 10))
    
    # Partial Dependence Plots
    n_features = min(len(feature_names), 9)
    for i in range(n_features):
        plt.subplot(3, 3, i + 1)
        
        try:
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            
            plt.plot(XX[:, i], pdep, 'b-', linewidth=2)
            plt.fill_between(XX[:, i], confi[:, 0], confi[:, 1], color='blue', alpha=0.1)
            plt.title(feature_names[i], fontweight='bold')
            plt.grid(True, alpha=0.2)
        except:
            plt.text(0.5, 0.5, "Error plotting", ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def run(file_path=None):
    """Run GAM pipeline."""
    try:
        X, y, feats = load_and_process_data(file_path, n_features=10)
        
        if len(y) < 20: 
            raise ValueError("Insufficient data.")
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled)

        gam = train_gam_automatic(X_train_scaled, y_train)

        acc = accuracy_score(y_test, gam.predict(X_test_scaled))
        try:
            probs = gam.predict_proba(X_test_scaled)
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0.0

        results = {
            'accuracy': acc,
            'auc': auc,
            'report': classification_report(y_test, gam.predict(X_test_scaled))
        }

        visualize_gam_interpretation(gam, scaler, feats, X_test_scaled, y_test)
        
        return results

    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    args = parser.parse_args()
    
    results = run(file_path=args.arquivo)
    if 'error' not in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
