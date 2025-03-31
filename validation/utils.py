import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from ..config import random_state as default_random_state
from matplotlib.patches import Patch

def perform_k_fold_cross_validation(X, y, model, n_splits=5, random_state=default_random_state, scoring='accuracy'):
    """
    Perform K-Fold cross-validation.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    model : estimator object
        The model to evaluate
    n_splits : int, default=5
        Number of folds
    random_state : int, default=None
        Controls the randomness of the CV splitter
    scoring : str or callable, default='accuracy'
        Scoring metric to use
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=kf, scoring=scoring, 
        return_train_score=True, return_estimator=True
    )
    
    return cv_results


def plot_cv_results(cv_results, title='Cross-Validation Results', figsize=(12, 6)):
    """
    Plot cross-validation results.
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results from cross_validate
    title : str, default='Cross-Validation Results'
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
    """
    # Extract metrics
    metrics = [key.replace('test_', '') for key in cv_results.keys() 
               if key.startswith('test_') and key != 'test_score']
    
    if not metrics:
        metrics = ['score']
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        train_scores = cv_results.get(f'train_{metric}', cv_results.get('train_score', []))
        test_scores = cv_results.get(f'test_{metric}', cv_results.get('test_score', []))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores)
        train_std = np.std(train_scores)
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        
        # Plot
        axes[i].bar([0, 1], [train_mean, test_mean], yerr=[train_std, test_std], 
                   capsize=10, color=['blue', 'orange'])
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Train', 'Test'])
        axes[i].set_ylabel(f'{metric.capitalize()} Score')
        axes[i].set_title(f'{metric.capitalize()} (Mean ± Std)')
        
        # Add text
        axes[i].text(0, train_mean, f'{train_mean:.4f}±{train_std:.4f}', 
                    ha='center', va='bottom')
        axes[i].text(1, test_mean, f'{test_mean:.4f}±{test_std:.4f}', 
                    ha='center', va='bottom')
    
    # Set title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def visualize_cv_splits(X, y, cv, groups=None, figsize=(12, 6)):
    """
    Visualize cross-validation splits.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    cv : cross-validation generator
        The cross-validation splitter to visualize
    groups : array-like, default=None
        Group labels for the samples
    figsize : tuple, default=(12, 6)
        Figure size
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get splits
    n_splits = getattr(cv, 'n_splits', 5)
    splits = list(cv.split(X, y, groups))
    
    # Create matrix to visualize splits
    n_samples = len(X)
    matrix = np.zeros((n_splits, n_samples))
    
    # Fill matrix
    for i, (train_idx, test_idx) in enumerate(splits):
        matrix[i, train_idx] = 1  # Training samples
        matrix[i, test_idx] = 2   # Test samples
    
    # Plot
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.viridis)
    plt.yticks(np.arange(n_splits), [f'Split {i+1}' for i in range(n_splits)])
    plt.xlabel('Sample index')
    
    # Add legend
    legend_elements = [
        Patch(facecolor=plt.cm.viridis(0.33), label='Training set'),
        Patch(facecolor=plt.cm.viridis(0.66), label='Test set')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2)
    
    plt.title('Cross-validation splits')
    plt.tight_layout()
    plt.show()