from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
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