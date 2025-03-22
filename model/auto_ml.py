import os
import h2o
import flaml
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
import pycaret.classification as pyc
from metric.classification_metric import ModelMetrics
from validation.validation import Validation
from model.utils import ModelUtils
warnings.filterwarnings('ignore')

class AutoMLBlender:
    """
    A comprehensive AutoML class for binary classification that leverages multiple AutoML frameworks:
    - H2O AutoML
    - Auto-sklearn
    - PyCaret
    - FLAML
    
    Features:
    - Automated model training with multiple AutoML frameworks
    - Model blending
    - Comprehensive evaluation
    - Model persistence
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
        """
        Initialize the AutoML blender.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.blended_model = None
        self.feature_names = None
        self.target_name = None
        self.class_names = None
        self.h2o_initialized = False
        
    def _init_h2o(self):
        """Initialize H2O if not already initialized."""
        if not self.h2o_initialized:
            h2o.init()
            self.h2o_initialized = True
    
    def fit(self, X, y, test_size=0.2, time_limit=3600, frameworks=None, validation_strategy='simple'):
        """
        Fit the AutoML models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The features
        y : array-like
            The target variable
        test_size : float, default=0.2
            The proportion of the dataset to include in the test split
        time_limit : int, default=3600
            Time limit in seconds for each framework
        frameworks : list, default=None
            List of frameworks to use. If None, all available frameworks are used.
            Options: 'h2o', 'pycaret', 'flaml'
        validation_strategy : str, default='simple'
            Validation strategy to use. Options: 'simple', 'stratified', 'time_series'
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store feature and target names
        self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        else:
            self.target_name = 'target'
        
        # Determine class names
        self.class_names = np.unique(y)
        
        # Split data based on validation strategy
        if validation_strategy == 'simple':
            X_train, X_test, y_train, y_test = Validation.train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        elif validation_strategy == 'stratified':
            X_train, X_test, y_train, y_test = Validation.train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        elif validation_strategy == 'time_series':
            X_train, X_test, y_train, y_test = Validation.time_series_train_test_split(
                X, y, test_size=test_size
            )
        else:
            raise ValueError(f"Unknown validation strategy: {validation_strategy}")
        
        # Determine which frameworks to use
        if frameworks is None:
            frameworks = ['h2o', 'pycaret', 'flaml']
        
        # Train models with each framework
        for framework in frameworks:
            if framework == 'h2o':
                self._train_h2o(X_train, y_train, X_test, y_test, time_limit)
            elif framework == 'pycaret':
                self._train_pycaret(X_train, y_train, X_test, y_test, time_limit)
            elif framework == 'flaml':
                self._train_flaml(X_train, y_train, X_test, y_test, time_limit)
            else:
                print(f"Warning: Unknown framework '{framework}'. Skipping.")
        
        # Create blended model
        if len(self.models) > 0:
            self._create_blended_model(X_test, y_test)
        
        return self
    
    def cross_validate(self, X, y, n_splits=5, time_limit=3600, frameworks=None, cv_strategy='kfold'):
        """
        Perform cross-validation with the AutoML models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The features
        y : array-like
            The target variable
        n_splits : int, default=5
            Number of folds for cross-validation
        time_limit : int, default=3600
            Time limit in seconds for each framework
        frameworks : list, default=None
            List of frameworks to use. If None, all available frameworks are used.
            Options: 'h2o', 'pycaret', 'flaml'
        cv_strategy : str, default='kfold'
            Cross-validation strategy to use. Options: 'kfold', 'stratified', 'time_series'
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Store feature and target names
        self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        else:
            self.target_name = 'target'
        
        # Determine class names
        self.class_names = np.unique(y)
        
        # Determine which frameworks to use
        if frameworks is None:
            frameworks = ['h2o', 'pycaret', 'flaml']
        
        # Initialize results dictionary
        cv_results = {framework: {'accuracy': [], 'precision': [], 'recall': [], 
                                 'f1': [], 'roc_auc': []} for framework in frameworks}
        
        # Perform cross-validation
        if cv_strategy == 'kfold':
            # Create KFold object
            kf = Validation.k_fold_cross_validation
        elif cv_strategy == 'stratified':
            # Create StratifiedKFold object
            kf = Validation.stratified_k_fold_cross_validation
        elif cv_strategy == 'time_series':
            # Create TimeSeriesSplit object
            kf = Validation.time_series_cross_validation
        else:
            raise ValueError(f"Unknown cross-validation strategy: {cv_strategy}")
        
        # Create a dummy model for cross-validation
        # We'll use the actual AutoML models in each fold
        dummy_model = type('DummyModel', (), {'fit': lambda *args, **kwargs: None, 
                                             'predict': lambda *args, **kwargs: None})()
        
        # Perform cross-validation
        cv_results = kf(X, y, dummy_model, n_splits=n_splits, random_state=self.random_state)
        
        # Extract train and test indices from each fold
        for i, (train_idx, test_idx) in enumerate(cv_results['cv_indices']):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train models with each framework
            for framework in frameworks:
                if framework == 'h2o':
                    self._train_h2o(X_train, y_train, X_test, y_test, time_limit)
                elif framework == 'pycaret':
                    self._train_pycaret(X_train, y_train, X_test, y_test, time_limit)
                elif framework == 'flaml':
                    self._train_flaml(X_train, y_train, X_test, y_test, time_limit)
                
                # Store metrics
                cv_results[framework]['accuracy'].append(self.models[framework]['accuracy'])
                cv_results[framework]['precision'].append(self.models[framework]['precision'])
                cv_results[framework]['recall'].append(self.models[framework]['recall'])
                cv_results[framework]['f1'].append(self.models[framework]['f1'])
                cv_results[framework]['roc_auc'].append(self.models[framework]['roc_auc'])
        
        # Calculate mean and std for each metric
        for framework in frameworks:
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                cv_results[framework][f'{metric}_mean'] = np.mean(cv_results[framework][metric])
                cv_results[framework][f'{metric}_std'] = np.std(cv_results[framework][metric])
        
        # Visualize cross-validation results
        Validation.plot_cv_results(cv_results)
        
        return cv_results
    
    def _train_h2o(self, X_train, y_train, X_test, y_test, time_limit):
        """Train models using H2O AutoML."""
        print("\n=== Training with H2O AutoML ===")
        start_time = time.time()
        
        # Initialize H2O
        self._init_h2o()
        
        # Combine features and target for H2O
        train_df = pd.concat([X_train.reset_index(drop=True), 
                             pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True), 
                            pd.DataFrame(y_test).reset_index(drop=True)], axis=1)
        
        # Convert to H2O frames
        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)
        
        # Set target and features
        target_col = self.target_name
        feature_cols = self.feature_names
        
        # Convert target to factor (categorical) for classification
        train_h2o[target_col] = train_h2o[target_col].asfactor()
        test_h2o[target_col] = test_h2o[target_col].asfactor()
        
        # Run AutoML
        aml = H2OAutoML(
            max_runtime_secs=time_limit,
            seed=self.random_state,
            nfolds=5,
            sort_metric="AUC"
        )
        
        aml.train(x=feature_cols, y=target_col, training_frame=train_h2o, validation_frame=test_h2o)
        
        # Get the best model
        best_model = aml.leader
        
        # Make predictions
        preds = best_model.predict(test_h2o)
        
        # Convert H2O predictions to numpy arrays
        y_pred = preds['predict'].as_data_frame().values.ravel()
        y_pred_proba = preds['p1'].as_data_frame().values.ravel()  # Probability of class 1
        
        # Calculate metrics using ModelMetrics
        metrics = ModelMetrics.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store the model
        self.models['h2o'] = {
            'model': best_model,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'framework': 'h2o'
        }
        
        # Print performance
        training_time = time.time() - start_time
        print(f"H2O AutoML trained in {training_time:.2f} seconds")
        print(f"Best model: {best_model.__class__.__name__}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def _train_pycaret(self, X_train, y_train, X_test, y_test, time_limit):
        """Train models using PyCaret."""
        print("\n=== Training with PyCaret ===")
        start_time = time.time()
        
        # Combine features and target for PyCaret
        train_df = pd.concat([X_train.reset_index(drop=True), 
                             pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
        
        # Setup PyCaret
        pyc.setup(
            data=train_df,
            target=self.target_name,
            session_id=self.random_state,
            silent=True,
            verbose=False
        )
        
        # Compare models
        best_model = pyc.compare_models(
            n_select=1,
            sort='AUC',
            verbose=False
        )
        
        # Finalize model
        final_model = pyc.finalize_model(best_model)
        
        # Make predictions
        y_pred = pyc.predict_model(final_model, data=X_test)['prediction_label'].values
        y_pred_proba = pyc.predict_model(final_model, data=X_test)['prediction_score'].values
        
        # Calculate metrics
        metrics = ModelMetrics.calculate_metrics(y_test, y_pred, y_pred_proba)

        
        # Store the model
        self.models['pycaret'] = {
            'model': final_model,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'framework': 'pycaret'
        }
        
        # Print performance
        training_time = time.time() - start_time
        print(f"PyCaret trained in {training_time:.2f} seconds")
        print(f"Best model: {type(final_model).__name__}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def _train_flaml(self, X_train, y_train, X_test, y_test, time_limit):
        """Train models using FLAML."""
        print("\n=== Training with FLAML ===")
        start_time = time.time()
        
        # Initialize FLAML
        automl = flaml.AutoML()
        
        # Fit FLAML
        automl.fit(
            X_train=X_train.values,
            y_train=y_train,
            task='classification',
            time_budget=time_limit,
            metric='roc_auc',
            seed=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Make predictions
        y_pred = automl.predict(X_test.values)
        y_pred_proba = automl.predict_proba(X_test.values)[:, 1]
        
        # Calculate metrics
        metrics = ModelMetrics.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Store the model
        self.models['flaml'] = {
            'model': automl,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'framework': 'flaml'
        }
        
        # Print performance
        training_time = time.time() - start_time
        print(f"FLAML trained in {training_time:.2f} seconds")
        print(f"Best model: {automl.best_estimator}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def _create_blended_model(self, X_test, y_test):
        """
        Create a blended model by averaging predictions from all models.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : array-like
            Test target
        """
        print("\n=== Creating Blended Model ===")
        
        # Get predictions from all models
        y_pred_probas = []
        for name, model_info in self.models.items():
            y_pred_probas.append(model_info['y_pred_proba'])
        
        # Average predictions
        y_pred_proba_blended = np.mean(y_pred_probas, axis=0)
        
        # Convert to binary predictions
        y_pred_blended = (y_pred_proba_blended > 0.5).astype(int)
        
        # Calculate metrics
        metrics = ModelMetrics.calculate_metrics(y_test, y_pred_blended, y_pred_proba_blended)

        
        # Store the blended model
        self.blended_model = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'y_pred': y_pred_blended,
            'y_pred_proba': y_pred_proba_blended,
            'models': list(self.models.keys())
        }
        
        print("Blended model created with the following frameworks:")
        for name in self.models.keys():
            print(f"- {name}")
        print(f"Blended model performance: Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def predict(self, X, framework=None):
        """
        Make predictions using a specific framework or the blended model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The input features
        framework : str, default=None
            The framework to use for predictions.
            If None, the blended model is used.
            
        Returns:
        --------
        array-like
            The predicted classes
        """
        if framework is not None and framework in self.models:
            return self._predict_with_framework(X, framework)
        elif self.blended_model:
            return self._predict_blended(X)
        else:
            # Use the best model based on ROC-AUC
            best_framework = max(self.models.items(), key=lambda x: x[1]['roc_auc'])[0]
            return self._predict_with_framework(X, best_framework)
    
    def _predict_with_framework(self, X, framework):
        """Make predictions using a specific framework."""
        model_info = self.models[framework]
        model = model_info['model']
        
        if framework == 'h2o':
            # Convert to H2O frame
            X_h2o = h2o.H2OFrame(X)
            preds = model.predict(X_h2o)
            return preds['predict'].as_data_frame().values.ravel()
        elif framework == 'pycaret':
            return pyc.predict_model(model, data=X)['prediction_label'].values
        else:  # flaml
            if framework == 'flaml':
                return model.predict(X.values)
            else:
                return model.predict(X)
    
    def _predict_blended(self, X):
        """Make predictions using the blended model."""
        # Get predictions from all models
        y_pred_probas = []
        for framework in self.blended_model['models']:
            if framework == 'h2o':
                # Convert to H2O frame
                X_h2o = h2o.H2OFrame(X)
                preds = self.models[framework]['model'].predict(X_h2o)
                y_pred_proba = preds['p1'].as_data_frame().values.ravel()
            elif framework == 'pycaret':
                y_pred_proba = pyc.predict_model(self.models[framework]['model'], data=X)['prediction_score'].values
            else:  # flaml
                if framework == 'flaml':
                    y_pred_proba = self.models[framework]['model'].predict_proba(X.values)[:, 1]
                else:
                    y_pred_proba = self.models[framework]['model'].predict_proba(X)[:, 1]
            
            y_pred_probas.append(y_pred_proba)
        
        # Average predictions
        y_pred_proba_blended = np.mean(y_pred_probas, axis=0)
        
        # Convert to binary predictions
        return (y_pred_proba_blended > 0.5).astype(int)
    
    def predict_proba(self, X, framework=None):
        """
        Make probability predictions using a specific framework or the blended model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The input features
        framework : str, default=None
            The framework to use for predictions.
            If None, the blended model is used.
            
        Returns:
        --------
        array-like
            The predicted probabilities for class 1
        """
        if framework is not None and framework in self.models:
            return self._predict_proba_with_framework(X, framework)
        elif self.blended_model:
            return self._predict_proba_blended(X)
        else:
            # Use the best model based on ROC-AUC
            best_framework = max(self.models.items(), key=lambda x: x[1]['roc_auc'])[0]
            return self._predict_proba_with_framework(X, best_framework)
    
    def _predict_proba_with_framework(self, X, framework):
        """Make probability predictions using a specific framework."""
        model_info = self.models[framework]
        model = model_info['model']
        
        if framework == 'h2o':
            # Convert to H2O frame
            X_h2o = h2o.H2OFrame(X)
            preds = model.predict(X_h2o)
            return preds['p1'].as_data_frame().values.ravel()
        elif framework == 'pycaret':
            return pyc.predict_model(model, data=X)['prediction_score'].values
        else:  # flaml
            if framework == 'flaml':
                return model.predict_proba(X.values)[:, 1]
            else:
                return model.predict_proba(X)[:, 1]
    
    def _predict_proba_blended(self, X):
        """Make probability predictions using the blended model."""
        # Get predictions from all models
        y_pred_probas = []
        for framework in self.blended_model['models']:
            if framework == 'h2o':
                # Convert to H2O frame
                X_h2o = h2o.H2OFrame(X)
                preds = self.models[framework]['model'].predict(X_h2o)
                y_pred_proba = preds['p1'].as_data_frame().values.ravel()
            elif framework == 'pycaret':
                y_pred_proba = pyc.predict_model(self.models[framework]['model'], data=X)['prediction_score'].values
            else:  # flaml
                if framework == 'flaml':
                    y_pred_proba = self.models[framework]['model'].predict_proba(X.values)[:, 1]
                else:
                    y_pred_proba = self.models[framework]['model'].predict_proba(X)[:, 1]
            
            y_pred_probas.append(y_pred_proba)
        
        # Average predictions
        return np.mean(y_pred_probas, axis=0)
    
    def evaluate(self, X_test, y_test, framework=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            The test features
        y_test : array-like
            The test target
        framework : str, default=None
            The framework to evaluate.
            If None, the blended model is evaluated.
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test, framework)
        y_pred_proba = self.predict_proba(X_test, framework)
        
        # Calculate metrics using ModelMetrics
        return ModelMetrics.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    def plot_roc_curve(self, X_test, y_test):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            The test features
        y_test : array-like
            The test target
        """
        y_pred_probas = []
        model_names = []
        
        for name, model_info in self.models.items():
            y_pred_probas.append(self.predict_proba(X_test, name))
            model_names.append(name)
        
        # Add blended model if available
        if self.blended_model:
            y_pred_probas.append(self.predict_proba(X_test))
            model_names.append('Blended Model')
        
        # Plot ROC curves
        ModelMetrics.plot_roc_curve(y_test, y_pred_probas, model_names)
    
    
    def plot_precision_recall_curve(self, X_test, y_test):
        """
        Plot Precision-Recall curves for all models.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            The test features
        y_test : array-like
            The test target
        """
        # Get predictions from all models
        y_pred_probas = []
        model_names = []
        
        for name, model_info in self.models.items():
            y_pred_probas.append(self.predict_proba(X_test, name))
            model_names.append(name)
        
        # Add blended model if available
        if self.blended_model:
            y_pred_probas.append(self.predict_proba(X_test))
            model_names.append('Blended Model')
        
        # Plot PR curves
        ModelMetrics.plot_precision_recall_curve(y_test, y_pred_probas, model_names)
    
    def plot_confusion_matrix(self, X_test, y_test, framework=None):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            The test features
        y_test : array-like
            The test target
        framework : str, default=None
            The framework to use.
            If None, the blended model is used.
        """
        # Make predictions
        y_pred = self.predict(X_test, framework)
        
        # Get title
        title = f"Confusion Matrix{' for ' + framework if framework else ' for Blended Model'}"
        
        # Plot confusion matrix
        ModelMetrics.plot_confusion_matrix(y_test, y_pred, self.class_names, title=title)
    
    def save_models(self, path):
        """
        Save all models to disk.
        
        Parameters:
        -----------
        path : str
            Path to save the models
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save each model
        for name, model_info in self.models.items():
            if name != 'h2o':  # H2O models are saved differently
                model_path = ModelUtils.save_model(
                    model_info['model'], 
                    path, 
                    f"{name}_model.pkl"
                )
                print(f"Saved {name} model to {model_path}")
            else:
                model_path = ModelUtils.save_h2o_model(
                    model_info['model'], 
                    path, 
                    f"{name}_model"
                )
                print(f"Saved {name} model to {model_path}")
        
        # Save blended model info
        if self.blended_model:
            blended_model_path = ModelUtils.save_model(
                self.blended_model, 
                path, 
                "blended_model_info.pkl"
            )
            print(f"Saved blended model info to {blended_model_path}")
        
        # Save class instance without the models
        instance_copy = self.__dict__.copy()
        instance_copy.pop('models', None)
        instance_copy.pop('blended_model', None)
        
        instance_path = ModelUtils.save_model(
            instance_copy, 
            path, 
            "automl_instance.pkl"
        )
        print(f"Saved AutoML instance info to {instance_path}")
    
    @classmethod
    def load_models(cls, path):
        """
        Load models from disk.
        
        Parameters:
        -----------
        path : str
            Path where models are saved
            
        Returns:
        --------
        AutoMLBlender
            Loaded AutoML instance
        """
        # Create a new instance
        instance = cls()
        
        # Load instance info
        instance_path = os.path.join(path, "automl_instance.pkl")
        if os.path.exists(instance_path):
            instance_info = ModelUtils.load_model(instance_path)
            for key, value in instance_info.items():
                setattr(instance, key, value)
        
        # Load models
        instance.models = {}
        
        # Check for H2O model
        h2o_model_path = os.path.join(path, "h2o_model")
        if os.path.exists(h2o_model_path):
            instance._init_h2o()
            h2o_model = ModelUtils.load_h2o_model(h2o_model_path)
            
            # We need to recreate the model info dictionary
            # This is a placeholder, actual metrics would need to be recalculated
            instance.models['h2o'] = {
                'model': h2o_model,
                'framework': 'h2o',
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'roc_auc': 0,
                'y_pred': None,
                'y_pred_proba': None
            }
        
        # Load other models
        for framework in ['pycaret', 'flaml']:
            model_path = os.path.join(path, f"{framework}_model.pkl")
            if os.path.exists(model_path):
                model = ModelUtils.load_model(model_path)
                
                # We need to recreate the model info dictionary
                instance.models[framework] = {
                    'model': model,
                    'framework': framework,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'roc_auc': 0,
                    'y_pred': None,
                    'y_pred_proba': None
                }
        
        # Load blended model info
        blended_model_path = os.path.join(path, "blended_model_info.pkl")
        if os.path.exists(blended_model_path):
            instance.blended_model = ModelUtils.load_model(blended_model_path)
        
        return instance
    
    def get_best_model(self):
        """
        Get the best model based on ROC-AUC score.
        
        Returns:
        --------
        tuple
            (framework_name, model_info)
        """
        if not self.models:
            return None
        
        return max(self.models.items(), key=lambda x: x[1]['roc_auc'])
    
    def get_model_comparison(self):
        """
        Get a comparison of all models.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with model comparison
        """
        if not self.models:
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison = []
        
        for name, model_info in self.models.items():
            comparison.append({
                'Framework': name,
                'Accuracy': model_info['accuracy'],
                'Precision': model_info['precision'],
                'Recall': model_info['recall'],
                'F1 Score': model_info['f1'],
                'ROC-AUC': model_info['roc_auc']
            })
        
        # Add blended model if available
        if self.blended_model:
            comparison.append({
                'Framework': 'Blended Model',
                'Accuracy': self.blended_model['accuracy'],
                'Precision': self.blended_model['precision'],
                'Recall': self.blended_model['recall'],
                'F1 Score': self.blended_model['f1'],
                'ROC-AUC': self.blended_model['roc_auc']
            })
        
        return pd.DataFrame(comparison).sort_values('ROC-AUC', ascending=False)
    
    def plot_model_comparison(self):
        """Plot a comparison of all models."""
        comparison_df = self.get_model_comparison()
        
        if comparison_df.empty:
            print("No models to compare.")
            return
        
        # Melt the dataframe for easier plotting
        melted_df = pd.melt(
            comparison_df, 
            id_vars=['Framework'], 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            var_name='Metric', 
            value_name='Value'
        )
        
        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Framework', y='Value', hue='Metric', data=melted_df)
        plt.title('Model Comparison')
        plt.xlabel('Framework')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def shutdown(self):
        """Shutdown H2O if it was initialized."""
        if self.h2o_initialized:
            h2o.shutdown()
            self.h2o_initialized = False