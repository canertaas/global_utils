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
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from ..config import random_state, classification_params, default_classification_settings
from ..validation.validation import Validation
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
    
    def __init__(self, scoring, validation_strategy=None, output_dir="model_outputs"): # todo impliment different validation approaches
        """
        Initialize ClassificationModels
        
        Parameters:
        -----------
        scoring : str, default=None
            Scoring metric for model evaluation
        validation_strategy : str, default='k_fold'
            Validation strategy to use
        validation_params : dict, default=None
            Parameters for the validation strategy
        output_dir : str, default="model_outputs"
            Directory to save model outputs and results
        """
        # Use default settings if not provided
        self.scoring = scoring or default_classification_settings['scoring']
        self.validation_strategy = validation_strategy
        
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
        
    def _validate_data(self, X, y):
        """Validate data based on validation strategy"""
        return Validation.train_test_split(
            X, y,
            test_size=default_classification_settings['test_size'],
            stratify=default_classification_settings['stratify'],
            random_state=random_state
        )
    
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
    
    def train_model_with_validation(self, model_name, X, y, optimize=True, 
                                  optimization_method=default_classification_settings['optimization_method']):
        """Train a model with validation and store detailed results"""
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Get feature names if X is DataFrame
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            
            # Split data if using train-test split
            X_train, X_val, y_train, y_val = self._validate_data(X, y)
    
            if optimize:
                try:
                    cv = Validation.get_stratified_k_fold(
                        n_splits=default_classification_settings['n_splits'],
                        random_state=random_state
                    )
                    
                    search = RandomizedSearchCV(
                        model,
                        param_distributions=model_info['params'],
                        n_iter=default_classification_settings['n_iter'],
                        cv=cv,
                        scoring=self.scoring,
                        random_state=random_state,
                        n_jobs=default_classification_settings['n_jobs'],
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
                    best_model.fit(X_train, y_train)  # Silinen kodu geri ekledim
                    cv_results = None
                    cv_scores = np.array([0.0])  # Varsayılan boş skor
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = "No optimization performed"
                cv_results = None
                cv_scores = Validation.stratified_k_fold_cross_validation(
                    model=best_model, X=X_train, y=y_train, 
                    n_splits=default_classification_settings['n_splits'],
                    random_state=random_state,
                    scoring=self.scoring
                )
                cv_scores = cv_scores['test_score']
            
            # Calculate metrics on validation set
            y_pred = best_model.predict(X_val if X_val is not None else X)
            y_true = y_val if y_val is not None else y
            
            # Get prediction probabilities for ROC curve if model supports it
            try:
                y_pred_proba = best_model.predict_proba(X_val if X_val is not None else X)
            except (AttributeError, ValueError, TypeError):
                # AttributeError: Model has no predict_proba method
                # ValueError: Wrong input shape or other value issues
                # TypeError: Incompatible types
                y_pred_proba = None
            
            metrics = ModelMetrics.calculate_metrics(y_true, y_pred)
            
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
            
            # Store detailed results
            self.model_results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'validation_metrics': metrics,
                'feature_names': feature_names
            }
            
            return best_model, cv_results, metrics
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
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
    
    def refit_with_full_data(self, X, y, model_names=None):
        """Refit selected models with full dataset"""
        if model_names is None:
            model_names = list(self.model_results.keys())
        
        for model_name in model_names:
            if model_name not in self.model_results:
                continue
                
            try:
                print(f"\nRefitting {model_name} with full dataset...")
                original_model = self.model_results[model_name]['model']
                
                # Clone and refit
                refitted_model = clone(original_model)
                refitted_model.fit(X, y)
                
                # Save the refitted model
                model_path = os.path.join(self.models_dir, f"{model_name}_final.joblib")
                joblib.dump(refitted_model, model_path)
                print(f"Saved refitted {model_name} to: {model_path}")
                
                # Update stored model
                self.model_results[model_name]['final_model'] = refitted_model
                
                # Get feature names
                feature_names = self.model_results[model_name].get('feature_names')
                
                # Generate model explanations for final model
                final_model_name = f"{model_name}_final"
                final_model_dir = os.path.join(self.results_dir, final_model_name)
                os.makedirs(final_model_dir, exist_ok=True)
                ModelExplainer.explain_model(
                    refitted_model, 
                    X, 
                    final_model_name, 
                    feature_names=feature_names, 
                    output_dir=final_model_dir
                )
                
            except Exception as e:
                print(f"Error refitting {model_name}: {str(e)}")

    def classification_runner(self, X, y, optimize=True, models_to_train=None, refit_final=True):
        """Runner function for the complete classification workflow"""
        
        # If the user does not provide a specific model list, use all available models by default
        if models_to_train is None:
            models_to_train = default_classification_settings['models_to_train']
        
        # Train models
        for model_name in models_to_train:
            print(f"Training {model_name}...")
            if model_name in self.models:  # Check if the model is valid
                self.train_model_with_validation(model_name, X, y, optimize=optimize)
            else:
                print(f"Warning: {model_name} is not a valid model name. Skipping.")
            print(f"Training {model_name} completed.")
        
        # Save results
        self.save_results_to_excel()
        
        # Refit final models with full dataset if requested
        if refit_final:
            print("\nRefitting final models with full dataset...")
            self.refit_with_full_data(X, y)
        
        return self
