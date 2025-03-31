from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from sklearn.calibration import calibration_curve

class ModelMetrics:
    """
    A utility class for calculating and visualizing model evaluation metrics
    for classification tasks.
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Calculate classification metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, default=None
            Predicted probabilities for the positive class
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        # Add ROC-AUC and Brier Score if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                
                # Add Brier Score for binary classification
                if len(np.unique(y_true)) == 2:
                    # Ensure we have probabilities for the positive class
                    if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
                        y_pred_proba_pos = y_pred_proba[:, 1]
                    else:
                        y_pred_proba_pos = y_pred_proba
                    
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba_pos)
            except Exception as e:
                print(f"Error calculating probability-based metrics: {str(e)}")
        
        return metrics
    
    @staticmethod
    def get_scorer(metric_name):
        """
        Get a scorer function for the specified metric.
        
        Parameters:
        -----------
        metric_name : str
            Name of the metric ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score')
            
        Returns:
        --------
        callable or str
            Scorer function or name for scikit-learn
        """
        from sklearn.metrics import make_scorer
        
        if metric_name == 'brier_score':
            # Note: we use negative brier_score because sklearn optimizes for higher values
            # but brier_score is better when lower
            return make_scorer(lambda y, y_pred: -brier_score_loss(y, y_pred), needs_proba=True)
        
        # For other metrics, return the string name that scikit-learn recognizes
        return metric_name
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), 
                             title='Confusion Matrix', save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, default=None
            List of class names
        figsize : tuple, default=(10, 8)
            Figure size
        title : str, default='Confusion Matrix'
            Plot title
        save_path : str, default=None
            If provided, save the plot to this path
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # If class names are not provided, use default names
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        # Plot confusion matrix
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        
        # Save or show
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_probas, model_names=None, figsize=(12, 8), 
                      save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_probas : list of array-like
            List of predicted probabilities for each model
        model_names : list, default=None
            List of model names
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, default=None
            If provided, save the plot to this path
        """
        plt.figure(figsize=figsize)
        
        # If model names are not provided, use default names
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(y_pred_probas))]
        
        # Plot ROC curve for each model
        for i, y_pred_proba in enumerate(y_pred_probas):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_names[i]} (AUC = {roc_auc:.4f})')
        
        # Plot random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save or show
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_probas, model_names=None, figsize=(12, 8),
                                   save_path=None):
        """
        Plot Precision-Recall curves for multiple models.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_probas : list of array-like
            List of predicted probabilities for each model
        model_names : list, default=None
            List of model names
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, default=None
            If provided, save the plot to this path
        """
        plt.figure(figsize=figsize)
        
        # If model names are not provided, use default names
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(y_pred_probas))]
        
        # Plot PR curve for each model
        for i, y_pred_proba in enumerate(y_pred_probas):
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, lw=2, label=f'{model_names[i]} (AUC = {pr_auc:.4f})')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)
        
        # Save or show
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def save_classification_report_visualization(y_true, y_pred, model_name, save_dir):
        """
        Generate and save visualization of the classification report.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model for titles and filenames
        save_dir : str
            Directory to save visualizations
        """
        try:
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate classification report as text
            report = classification_report(y_true, y_pred)
            
            # Save text report
            with open(os.path.join(save_dir, f'{model_name}_classification_report.txt'), 'w') as f:
                f.write(report)
            
            # Generate confusion matrix visualization
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Convert classification report to DataFrame for visualization
            try:
                # Parse the text report
                report_data = []
                lines = report.split('\n')
                for line in lines[2:-3]:  # Skip header and footer
                    if not line:
                        continue
                    row_data = line.strip().split()
                    if len(row_data) < 5:  # Skip rows without enough data
                        continue
                    class_name = row_data[0]
                    precision = float(row_data[1])
                    recall = float(row_data[2])
                    f1_score = float(row_data[3])
                    support = int(row_data[4])
                    report_data.append([class_name, precision, recall, f1_score, support])
                
                df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
                
                # Visualize metrics as a grouped bar chart
                df_plot = df.melt(id_vars=['Class', 'Support'], 
                                 value_vars=['Precision', 'Recall', 'F1-Score'],
                                 var_name='Metric', value_name='Value')
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Class', y='Value', hue='Metric', data=df_plot)
                plt.title(f'Classification Metrics for {model_name}')
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{model_name}_classification_metrics.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error visualizing classification report: {str(e)}")
                
        except Exception as e:
            print(f"Error in save_classification_report_visualization: {str(e)}")
    
    @staticmethod
    def evaluate_calibration(y_true, y_pred_proba, n_bins=10):
        """
        Evaluate the calibration of probability predictions.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred_proba : array-like
            Probability estimates for the positive class
        n_bins : int, default=10
            Number of bins for calibration curve
            
        Returns:
        --------
        dict
            Dictionary with calibration metrics
        """
        try:
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
            
            # Calculate mean absolute calibration error
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            
            # Calculate Brier score
            brier_score = brier_score_loss(y_true, y_pred_proba)
            
            return {
                'calibration_curve': {
                    'prob_true': prob_true,
                    'prob_pred': prob_pred
                },
                'calibration_error': calibration_error,
                'brier_score': brier_score
            }
            
        except Exception as e:
            print(f"Error evaluating calibration: {str(e)}")
            return None