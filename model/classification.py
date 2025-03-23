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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from ..config import random_state, classification_params
from ..validation.validation import Validation
from ..metric.classification_metric import ModelMetrics
from ..utils import create_output_directories


class ClassificationModels:
    """
    A comprehensive class for training and evaluating various classification models.
    Includes:
    - Multiple classification algorithms
    - Hyperparameter optimization
    - Model evaluation
    - Model comparison
    """
    
    def __init__(self, scoring, validation_strategy='k_fold', validation_params=None, output_dir="model_outputs"):
        """
        Initialize ClassificationModels
        
        Parameters:
        -----------
        scoring : str
            Scoring metric for model evaluation
        validation_strategy : str, default='k_fold'
            Validation strategy to use:
            - 'k_fold': K-Fold cross validation
            - 'stratified_k_fold': Stratified K-Fold cross validation
            - 'group_k_fold': Group K-Fold cross validation
            - 'time_series': Time series cross validation
            - 'train_test_split': Simple train-test split
        validation_params : dict, default=None
            Parameters for the validation strategy:
            - k_fold: {'n_splits': 5}
            - stratified_k_fold: {'n_splits': 5}
            - group_k_fold: {'n_splits': 5, 'groups': array-like}
            - time_series: {'n_splits': 5}
            - train_test_split: {'test_size': 0.2, 'stratify': None}
        output_dir : str, default="model_outputs"
            Directory to save model outputs and results
        """
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
                'model': LGBMClassifier(random_state=random_state),
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
        
        self.scoring = scoring
        self.validation_strategy = validation_strategy
        self.validation_params = validation_params or self._get_default_validation_params()
        self.fitted_models = {}
        self.best_model = None
        self.best_score = 0
        self.output_dir = output_dir
        self.results_dir = None
        self.model_results = {}  
        create_output_directories()
        
    def _get_default_validation_params(self):
        """Get default parameters for validation strategy"""
        defaults = {
            'k_fold': {'n_splits': 5},
            'stratified_k_fold': {'n_splits': 5},
            'group_k_fold': {'n_splits': 5},
            'time_series': {'n_splits': 5},
            'train_test_split': {'test_size': 0.2, 'stratify': None}
        }
        return defaults.get(self.validation_strategy, {'n_splits': 5})
    
    def _validate_data(self, X, y):
        """Validate data based on validation strategy"""
        if self.validation_strategy == 'train_test_split':
            return Validation.train_test_split(
                X, y,
                test_size=self.validation_params.get('test_size', 0.2),
                stratify=self.validation_params.get('stratify'),
                random_state=random_state
            )
        return X, None, y, None
    
    def train_model_with_validation(self, model_name, X, y, optimize=True, 
                                  optimization_method='random', n_iter=50):
        """Train a model with validation and store detailed results"""
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Split data if using train-test split
            X_train, X_val, y_train, y_val = self._validate_data(X, y)
            
            if optimize:
                try:
                    search = RandomizedSearchCV(
                        model, model_info['params'],
                        n_iter=n_iter,
                        cv=5 if self.validation_strategy != 'train_test_split' else None,
                        scoring=self.scoring,
                        random_state=random_state
                    ) if optimization_method == 'random' else GridSearchCV(
                        model, model_info['params'],
                        cv=5 if self.validation_strategy != 'train_test_split' else None,
                        scoring=self.scoring
                    )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    print(f"\nBest parameters for {model_name}:")
                    print(best_params)
                except Exception as e:
                    print(f"Optimization failed for {model_name}: {str(e)}")
                    print("Training with default parameters...")
                    best_model = model
                    best_params = "Failed to optimize"
                    best_model.fit(X_train, y_train)
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = "No optimization performed"
            
            # Validation scores
            if self.validation_strategy == 'train_test_split':
                val_score = best_model.score(X_val, y_val)
                cv_results = {'test_score': np.array([val_score])}
                cv_scores = [val_score]
            else:
                validation_method = getattr(Validation, f'{self.validation_strategy}_cross_validation')
                cv_results = validation_method(
                    X, y, best_model,
                    **self.validation_params,
                    random_state=random_state,
                    scoring=self.scoring
                )
                cv_scores = cv_results['test_score']
            
            # Calculate metrics on validation set
            y_pred = best_model.predict(X_val if X_val is not None else X)
            y_true = y_val if y_val is not None else y
            metrics = ModelMetrics.calculate_metrics(y_true, y_pred)
            
            # Calculate feature importance
            try:
                if hasattr(best_model, 'feature_importances_'):
                    importance = best_model.feature_importances_
                else:
                    # Use permutation importance for models without feature_importances_
                    perm_importance = permutation_importance(best_model, X_val if X_val is not None else X,
                                                          y_val if y_val is not None else y,
                                                          n_repeats=5, random_state=random_state)
                    importance = perm_importance.importances_mean
            except Exception as e:
                print(f"Could not calculate feature importance for {model_name}: {str(e)}")
                importance = None
            
            # Store detailed results
            self.model_results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'validation_metrics': metrics,
                'feature_importance': importance
            }
            
            return best_model, cv_results, metrics
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return None, None, None
    
    def save_results_to_excel(self):
        """Save all results to an Excel file with multiple sheets"""
        output_path = os.path.join(self.results_dir, "model_results.xlsx")
        
        # Model performance summary
        summary_data = []
        for model_name, results in self.model_results.items():
            summary_data.append({
                'Model': model_name,
                'CV Mean Score': results['cv_mean'],
                'CV Std': results['cv_std'],
                **results['validation_metrics']
            })
        summary_df = pd.DataFrame(summary_data)
        
        # CV scores for each model
        cv_scores_data = {}
        for model_name, results in self.model_results.items():
            cv_scores_data[model_name] = results['cv_scores']
        cv_scores_df = pd.DataFrame(cv_scores_data)
        
        # Feature importance for each model
        feature_importance_data = []
        for model_name, results in self.model_results.items():
            if results['feature_importance'] is not None:
                importance_dict = {
                    'Model': model_name,
                    'Feature': range(len(results['feature_importance'])),
                    'Importance': results['feature_importance']
                }
                feature_importance_data.append(pd.DataFrame(importance_dict))
        if feature_importance_data:
            feature_importance_df = pd.concat(feature_importance_data, ignore_index=True)
        else:
            feature_importance_df = pd.DataFrame()
        
        # Best parameters for each model
        params_data = []
        for model_name, results in self.model_results.items():
            params_data.append({
                'Model': model_name,
                'Best Parameters': str(results['best_params'])
            })
        params_df = pd.DataFrame(params_data)
        
        # Save all to Excel
        with pd.ExcelWriter(output_path) as writer:
            summary_df.to_excel(writer, sheet_name='Model Summary', index=False)
            cv_scores_df.to_excel(writer, sheet_name='CV Scores', index=False)
            if not feature_importance_df.empty:
                feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
            params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        
        print(f"\nResults saved to: {output_path}")
        
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
                model_path = os.path.join(self.results_dir, "models", f"{model_name}_final.joblib")
                joblib.dump(refitted_model, model_path)
                print(f"Saved refitted {model_name} to: {model_path}")
                
                # Update stored model
                self.model_results[model_name]['final_model'] = refitted_model
                
            except Exception as e:
                print(f"Error refitting {model_name}: {str(e)}")

def classification_runner(X, y, scoring, validation_strategy='k_fold', validation_params=None,
                        optimize=True, models_to_train=None, refit_final=True, 
                        output_dir="model_outputs"):
    """Runner function for the complete classification workflow"""
    
    # Initialize classifier
    classifier = ClassificationModels(
        scoring=scoring,
        validation_strategy=validation_strategy,
        validation_params=validation_params,
        output_dir=output_dir
    )
    
    # Train models
    if models_to_train is not None:
        for model_name in models_to_train:
            classifier.train_model_with_validation(model_name, X, y, optimize=optimize)
    else:
        for model_name in classifier.models.keys():
            classifier.train_model_with_validation(model_name, X, y, optimize=optimize)
    
    # Save results
    classifier.save_results_to_excel()
    
    # Refit final models with full dataset if requested
    if refit_final:
        print("\nRefitting final models with full dataset...")
        classifier.refit_with_full_data(X, y)
    
    return classifier

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Load example dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Run classification with specific models
    results = classification_runner(
        X_train, X_test, y_train, y_test,
        models_to_train=['logistic', 'random_forest', 'xgboost'],
        optimize=True,
        validation_strategy='k_fold',
        validation_params=None,
        cv=5,
        n_iter=20
    )
    
    # Access results
    best_model = results['best_model']
    comparison = results['comparison']
    classifier = results['classifier']
