from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

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
    def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), title='Confusion Matrix'):
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
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_probas, model_names=None, figsize=(12, 8)):
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
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_probas, model_names=None, figsize=(12, 8)):
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
        plt.show()
    
    @staticmethod
    def print_classification_report(y_true, y_pred, target_names=None):
        """
        Print classification report.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        target_names : list, default=None
            List of target class names
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("Classification Report:")
        print(report)