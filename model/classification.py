import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from ..config import random_state, classification_params, default_classification_settings
from ..validation.get_validation import GetValidation
from ..metric.classification_metric import ModelMetrics
from ..utils import create_output_directories
from ..model.explain import ModelExplainer


class ClassificationModels:
    """
    A comprehensive class for training and evaluating various classification models.
    Includes:
    - Multiple classification algorithms
    - Hyperparameter optimization
    - Model evaluation
    - Model comparison
    """
    
    def __init__(self, 
                 scoring=None, 
                 validation_method='stratified_k_fold',
                 validation_params=None, 
                 output_dir="model_outputs"):
        """
        Initialize ClassificationModels
        
        Parameters:
        -----------
        scoring : str, default=None
            Scoring metric for model evaluation
        validation_method : str, default='stratified_k_fold'
            Validation method to use. Options:
            - 'train_test_split': Simple train-test split
            - 'stratified_k_fold': Stratified K-Fold cross-validation
            - 'k_fold': K-Fold cross-validation
            - 'group_k_fold': Group K-Fold cross-validation 
            - 'time_series': Time Series cross-validation
        validation_params : dict, default=None
            Parameters for the validation method
        output_dir : str, default="model_outputs"
            Directory to save model outputs and results
        """
        # Use default settings if not provided, and get the appropriate scorer from ModelMetrics
        self.scoring = ModelMetrics.get_scorer(scoring) or ModelMetrics.get_scorer(default_classification_settings['scoring'])
        
        # Set validation method and parameters
        self.validation_method = validation_method
        self.validation_params = validation_params or {}
        
        # Use default parameters from settings if not provided
        for param, default_value in default_classification_settings.items():
            if param not in self.validation_params:
                self.validation_params[param] = default_value
        
        # Create time-stamped run directory
        self.output_dir = output_dir
        self.results_dir = create_output_directories(output_dir)
        
        # Create models directory for saving model files
        self.models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {
            'logistic': {
                'model': LogisticRegression(random_state=random_state),
                'params': classification_params['logistic']
            },
            'svm': {
                'model': SVC(random_state=random_state, probability=True),
                'params': classification_params['svm']
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': classification_params['random_forest']
            },
            'xgboost': {
                'model': XGBClassifier(random_state=random_state),
                'params': classification_params['xgboost']
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=random_state, verbose=0),
                'params': classification_params['lightgbm']
            },
            'catboost': {
                'model': CatBoostClassifier(random_state=random_state, verbose=False),
                'params': classification_params['catboost']
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=random_state),
                'params': classification_params['decision_tree']
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': classification_params['knn']
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': classification_params['naive_bayes']
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': classification_params['gradient_boosting']
            }
        }
        
        self.fitted_models = {}
        self.best_model = None
        self.best_score = 0
        self.model_results = {}  
        
    def save_classification_report(self, y_true, y_pred, model_name):
        """Save classification report as a visualization"""
        # Create model-specific directory for results
        model_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Use ModelMetrics to save report visualization
        ModelMetrics.save_classification_report_visualization(y_true, y_pred, model_name, model_dir)
    
    def save_roc_curve(self, y_true, y_pred_proba, model_name):
        """Save ROC curve for binary classification"""
        # Create model-specific directory for results
        model_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                # Get probabilities for the positive class
                if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                
                # Use ModelMetrics to save ROC curve
                save_path = os.path.join(model_dir, f'{model_name}_roc_curve.png')
                ModelMetrics.plot_roc_curve(y_true, [y_pred_proba], [model_name], 
                                          figsize=(10, 8), save_path=save_path)
            else:
                # For multi-class, implement later if needed
                print(f"Multi-class ROC not implemented for {model_name}")
        
        except Exception as e:
            print(f"Error saving ROC curve for {model_name}: {str(e)}")
    
    def train_model_with_validation(self, model_name, X, y, groups=None, optimize=True):
        """
        Train a model with validation and store detailed results
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X : array-like
            Features
        y : array-like
            Target variable
        groups : array-like, default=None
            Group labels for samples (for group-based validation methods)
        optimize : bool, default=True
            Whether to optimize hyperparameters
            
        Returns:
        --------
        best_model, cv_results, metrics : tuple
            The trained model, cross-validation results, and validation metrics
        """
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Get feature names if X is DataFrame
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            
            # Split data based on chosen validation strategy using GetValidation
            X_train, X_val, y_train, y_val = GetValidation.validate_data(
                X, y, groups, 
                validation_method=self.validation_method,
                validation_params=self.validation_params
            )
    
            if optimize:
                try:
                    # Get cross-validator based on validation method using GetValidation
                    cv = GetValidation.get_cross_validator(
                        validation_method=self.validation_method,
                        validation_params=self.validation_params,
                        X=X, y=y, groups=groups
                    )
                    
                    search = RandomizedSearchCV(
                        model,
                        param_distributions=model_info['params'],
                        n_iter=self.validation_params.get('n_iter', 10),
                        cv=cv,
                        scoring=self.scoring,
                        random_state=random_state,
                        n_jobs=self.validation_params.get('n_jobs', -1),
                        error_score='raise',
                        return_train_score=True
                    )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    cv_results = search.cv_results_
                    cv_scores = cv_results['mean_test_score']
                    
                except Exception as e:
                    print(f"Optimization failed for {model_name}: {str(e)}")
                    best_model = model
                    best_params = "Failed to optimize"
                    best_model.fit(X_train, y_train)
                    cv_results = None
                    cv_scores = np.array([0.0])  # Default empty score
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = "No optimization performed"
                
                # Perform cross-validation based on validation method using GetValidation
                cv_results = GetValidation.perform_cross_validation(
                    model=best_model, 
                    X=X_train, 
                    y=y_train, 
                    groups=groups,
                    validation_method=self.validation_method,
                    validation_params=self.validation_params,
                    scoring=self.scoring
                )
                
                # Extract scores from CV results
                if isinstance(cv_results, dict) and 'test_score' in cv_results:
                    cv_scores = cv_results['test_score']
                else:
                    cv_scores = np.array([0.0])  # Fallback if CV fails
            
            # Calculate metrics on validation set
            y_pred = best_model.predict(X_val if X_val is not None else X)
            y_true = y_val if y_val is not None else y
            
            # Get prediction probabilities for ROC curve if model supports it
            try:
                y_pred_proba = best_model.predict_proba(X_val if X_val is not None else X)
            except (AttributeError, ValueError, TypeError):
                y_pred_proba = None
            
            metrics = ModelMetrics.calculate_metrics(y_true, y_pred, y_pred_proba[:, 1] if y_pred_proba is not None and y_pred_proba.ndim > 1 else y_pred_proba)
            
            # Store detailed results - do this BEFORE calibration
            self.model_results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'validation_metrics': metrics,
                'feature_names': feature_names
            }
            
            # Create model-specific directory for results
            model_dir = os.path.join(self.results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Generate model explanations using ModelExplainer
            print(f"\nGenerating model explanations for {model_name}...")
            ModelExplainer.explain_model(
                best_model, 
                X_val if X_val is not None else X, 
                model_name, 
                feature_names=feature_names, 
                output_dir=model_dir
            )
            
            # Save ROC curve if applicable
            if y_pred_proba is not None:
                self.save_roc_curve(y_true, y_pred_proba, model_name)
            
            # Save classification report visualization
            self.save_classification_report(y_true, y_pred, model_name)
            
            # Save the final model (calibrated or not)
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(best_model, model_path)
            print(f"Saved {model_name} to: {model_path}")
            
            return best_model, cv_results, metrics
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
            return None, None, None
    
    def save_results_to_excel(self):
        """Save model results to Excel file"""
        # Prepare summary data
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        auc_scores = []
        cv_means = []
        cv_stds = []
        
        for model_name, results in self.model_results.items():
            metrics = results.get('validation_metrics', {})
            models.append(model_name)
            accuracies.append(metrics.get('accuracy', float('nan')))
            precisions.append(metrics.get('precision', float('nan')))
            recalls.append(metrics.get('recall', float('nan')))
            f1s.append(metrics.get('f1', float('nan')))
            auc_scores.append(metrics.get('roc_auc', float('nan')))
            cv_means.append(results.get('cv_mean', float('nan')))
            cv_stds.append(results.get('cv_std', float('nan')))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1s,
            'ROC AUC': auc_scores,
            'CV Mean Score': cv_means,
            'CV Std Dev': cv_stds
        })
        
        # Save to Excel
        excel_path = os.path.join(self.results_dir, "model_results_summary.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Model results saved to: {excel_path}")

        # Also save as CSV
        csv_path = os.path.join(self.results_dir, "model_results_summary.csv")
        df.to_csv(csv_path, index=False)
        
        return df

    def classification_runner(self, X, y, groups=None, optimize=True, models_to_train=None):
        """
        Runner function for the complete classification workflow
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        groups : array-like, default=None
            Group labels for samples (for group-based validation methods)
        optimize : bool, default=True
            Whether to optimize hyperparameters
        models_to_train : list, default=None
            List of model names to train. If None, uses default from settings
            
        Returns:
        --------
        self : ClassificationModels
            Returns self instance for method chaining
        """
        
        # If the user does not provide a specific model list, use all available models by default
        if models_to_train is None:
            models_to_train = default_classification_settings['models_to_train']
        
        # Train models
        for model_name in models_to_train:
            print(f"Training {model_name}...")
            if model_name in self.models:  # Check if the model is valid
                self.train_model_with_validation(model_name, X, y, groups, optimize=optimize   )
            else:
                print(f"Warning: {model_name} is not a valid model name. Skipping.")
            print(f"Training {model_name} completed.")
        
        # Save results
        self.save_results_to_excel()
        
        return self
