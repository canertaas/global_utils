import os
import joblib
import h2o
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ModelUtils:
    """
    Utility class for model persistence operations like saving and loading models.
    """
    
    @staticmethod
    def save_model(model, path, model_name):
        """
        Save a model to disk.
        
        Parameters:
        -----------
        model : object
            The model to save
        path : str
            Directory path to save the model
        model_name : str
            Name of the model file
            
        Returns:
        --------
        str
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Determine full path
        full_path = os.path.join(path, model_name)
        
        # Save the model
        joblib.dump(model, full_path)
        
        return full_path
    
    @staticmethod
    def load_model(path):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        path : str
            Path to the model file
            
        Returns:
        --------
        object
            The loaded model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        return joblib.load(path)
    
    @staticmethod
    def save_h2o_model(model, path, model_name):
        """
        Save an H2O model to disk.
        
        Parameters:
        -----------
        model : h2o.model
            The H2O model to save
        path : str
            Directory path to save the model
        model_name : str
            Name of the model directory
            
        Returns:
        --------
        str
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Determine full path
        full_path = os.path.join(path, model_name)
        
        # Save the model
        h2o.save_model(model, path=full_path)
        
        return full_path
    
    @staticmethod
    def load_h2o_model(path):
        """
        Load an H2O model from disk.
        
        Parameters:
        -----------
        path : str
            Path to the H2O model directory
            
        Returns:
        --------
        h2o.model
            The loaded H2O model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"H2O model directory not found at {path}")
        
        return h2o.load_model(path)


def plot_calibration_comparison(y_true, y_pred_probas, labels, model_name, save_path):
    """
    Plot calibration curves before and after calibration.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_probas : list of array-like
        List of predicted probabilities (before and after calibration)
    labels : list of str
        Labels for the curves
    model_name : str
        Name of the model
    save_path : str
        Path to save the plot
    """
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 8))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot calibration curves
    for i, y_pred_proba in enumerate(y_pred_probas):
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=labels[i])
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_calibration_metrics(metrics_before, metrics_after, model_name, save_dir):
    """
    Save calibration improvement metrics.
    
    Parameters:
    -----------
    metrics_before : dict
        Metrics before calibration
    metrics_after : dict
        Metrics after calibration
    model_name : str
        Name of the model
    save_dir : str
        Directory to save results
    """
    # Create a comparison table
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score']
    
    comparison = {
        'Metric': [],
        'Before Calibration': [],
        'After Calibration': [],
        'Improvement': []
    }
    
    for metric in metrics_to_compare:
        if metric in metrics_before and metric in metrics_after:
            before_val = metrics_before[metric]
            after_val = metrics_after[metric]
            
            # For brier_score, lower is better
            if metric == 'brier_score':
                improvement = before_val - after_val
            else:
                improvement = after_val - before_val
            
            comparison['Metric'].append(metric)
            comparison['Before Calibration'].append(before_val)
            comparison['After Calibration'].append(after_val)
            comparison['Improvement'].append(improvement)
    
    # Create DataFrame and save
    df = pd.DataFrame(comparison)
    df.to_csv(os.path.join(save_dir, f"{model_name}_calibration_comparison.csv"), index=False)
    
    # Also create a bar chart of improvements
    plt.figure(figsize=(12, 8))
    
    # For visualization, we'll use absolute values for brier_score improvement
    plot_data = df.copy()
    plot_data.loc[plot_data['Metric'] == 'brier_score', 'Improvement'] = abs(
        plot_data.loc[plot_data['Metric'] == 'brier_score', 'Improvement']
    )
    
    colors = ['green' if val >= 0 else 'red' for val in plot_data['Improvement']]
    
    plt.bar(plot_data['Metric'], plot_data['Improvement'], color=colors)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f'Calibration Improvement - {model_name}')
    plt.ylabel('Improvement')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(plot_data['Improvement']):
        plt.text(i, v + 0.01 if v >= 0 else v - 0.03, 
                f"{v:.4f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_calibration_improvement.png"), dpi=300)
    plt.close()