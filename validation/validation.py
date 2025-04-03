import numpy as np
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, GroupKFold, 
                                    TimeSeriesSplit, cross_validate, cross_val_score)
from ..config import random_state as default_random_state
from .utils import plot_cv_results, visualize_cv_splits

class Validation:
    """
    A comprehensive validation utility class that provides various cross-validation
    strategies and train-test splitting functionality for machine learning models.
    
    This class supports:
    - Simple train-test splits
    - K-Fold cross-validation
    - Stratified K-Fold cross-validation
    - Group K-Fold cross-validation
    - Time Series cross-validation
    - Custom validation strategies
    - Combinatorial Purged Group K-Fold cross-validation
    
    It also provides visualization tools for cross-validation results.
    """
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=default_random_state, stratify=None):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=None
            Controls the shuffling applied to the data before applying the split
        stratify : array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            The split data
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    
    @staticmethod
    def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=default_random_state, stratify=None):
        """
        Split the data into training, validation, and testing sets.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        val_size : float, default=0.2
            Proportion of the dataset to include in the validation split
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=None
            Controls the shuffling applied to the data before applying the split
        stratify : array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels
            
        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test : arrays
            The split data
        """
        # First, split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify if stratify is not None else y
        )
        
        # Then split train+val into train and val
        # Adjust val_size to account for the test split
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=y_train_val if stratify is not None else y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def group_train_test_split(X, y, groups, test_size=0.2, random_state=default_random_state):
        """
        Split the data into training and testing sets based on groups.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        groups : array-like
            Group labels for the samples
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=None
            Controls the shuffling applied to the data before applying the split
            
        Returns:
        --------
        X_train, X_test, y_train, y_test, groups_train, groups_test : arrays
            The split data and corresponding groups
        """
        # Get unique groups
        unique_groups = np.unique(groups)
        
        # Shuffle groups
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(unique_groups)
        
        # Split groups
        n_test_groups = int(len(unique_groups) * test_size)
        test_groups = unique_groups[:n_test_groups]
        train_groups = unique_groups[n_test_groups:]
        
        # Create masks
        train_mask = np.isin(groups, train_groups)
        test_mask = np.isin(groups, test_groups)
        
        # Split data
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train, groups_test = groups[train_mask], groups[test_mask]
        
        return X_train, X_test, y_train, y_test, groups_train, groups_test
    
    @staticmethod
    def time_series_train_test_split(X, y, test_size=0.2):
        """
        Split the data into training and testing sets for time series data.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            The split data
        """
        # Calculate split point
        split_idx = int(len(X) * (1 - test_size))
        
        # Split data
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def k_fold_cross_validation(X, y, model=None, n_splits=5, random_state=default_random_state, 
                               scoring=None, return_estimator=False, return_train_score=True):
        """
        Create a KFold cross-validator or perform K-Fold cross-validation.
        
        Parameters:
        -----------
        X : array-like or None
            Features. If None, returns only the cross-validator object.
        y : array-like or None
            Target variable. If None, returns only the cross-validator object.
        model : estimator object, default=None
            The model to evaluate. If None, returns only the cross-validator object.
        n_splits : int, default=5
            Number of folds
        random_state : int, default=None
            Controls the randomness of the CV splitter
        scoring : str, callable, or None, default=None
            Scoring metric to use
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
            
        Returns:
        --------
        If model is None:
            cv : KFold object
                The KFold cross-validator
        Else:
            results : dict
                Cross-validation results
        """
        # Create KFold object
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # If no model provided, return the cv object
        if model is None:
            return cv
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, 
            return_train_score=return_train_score,
            return_estimator=return_estimator
        )
        
        return cv_results
    
    @staticmethod
    def stratified_k_fold_cross_validation(X, y, model=None, n_splits=5, random_state=default_random_state, 
                                         scoring=None, return_estimator=False, return_train_score=True):
        """
        Create a StratifiedKFold cross-validator or perform stratified k-fold cross-validation.
        
        Parameters:
        -----------
        X : array-like or None
            Features. If None, returns only the cross-validator object.
        y : array-like or None
            Target variable. If None, returns only the cross-validator object.
        model : estimator object, default=None
            The model to evaluate. If None, returns only the cross-validator object.
        n_splits : int, default=5
            Number of folds
        random_state : int, default=None
            Controls the randomness of the CV splitter
        scoring : str, callable, or None, default=None
            Scoring metric to use
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
            
        Returns:
        --------
        If model is None:
            cv : StratifiedKFold object
                The StratifiedKFold cross-validator
        Else:
            results : dict
                Cross-validation results
        """
        # Create StratifiedKFold object
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # If no model provided, return the cv object
        if model is None or X is None or y is None:
            return cv
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, 
            return_train_score=return_train_score,
            return_estimator=return_estimator
        )
        
        return cv_results
    
    @staticmethod
    def group_k_fold_cross_validation(X, y, groups, model=None, n_splits=5, 
                                     scoring=None, return_estimator=False, return_train_score=True):
        """
        Create a GroupKFold cross-validator or perform Group K-Fold cross-validation.
        
        Parameters:
        -----------
        X : array-like or None
            Features. If None, returns only the cross-validator object.
        y : array-like or None
            Target variable. If None, returns only the cross-validator object.
        groups : array-like
            Group labels for the samples
        model : estimator object, default=None
            The model to evaluate. If None, returns only the cross-validator object.
        n_splits : int, default=5
            Number of folds
        scoring : str, callable, or None, default=None
            Scoring metric to use
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
            
        Returns:
        --------
        If model is None:
            cv : GroupKFold object
                The GroupKFold cross-validator
        Else:
            results : dict
                Cross-validation results
        """
        # Create GroupKFold object
        cv = GroupKFold(n_splits=n_splits)
        
        # If no model provided, return the cv object
        if model is None or X is None or y is None:
            return cv
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, groups=groups, scoring=scoring, 
            return_train_score=return_train_score,
            return_estimator=return_estimator
        )
        
        return cv_results
    
    @staticmethod
    def time_series_cross_validation(X, y, model=None, n_splits=5, test_size=None, gap=0, 
                                    scoring=None, return_estimator=False, return_train_score=True):
        """
        Create a TimeSeriesSplit cross-validator or perform Time Series cross-validation.
        
        Parameters:
        -----------
        X : array-like or None
            Features. If None, returns only the cross-validator object.
        y : array-like or None
            Target variable. If None, returns only the cross-validator object.
        model : estimator object, default=None
            The model to evaluate. If None, returns only the cross-validator object.
        n_splits : int, default=5
            Number of splits
        test_size : int, default=None
            Number of samples in each test set
        gap : int, default=0
            Number of samples to exclude from the end of each train set before the test set
        scoring : str, callable, or None, default=None
            Scoring metric to use
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
            
        Returns:
        --------
        If model is None:
            cv : TimeSeriesSplit object
                The TimeSeriesSplit cross-validator
        Else:
            results : dict
                Cross-validation results
        """
        # Create TimeSeriesSplit object
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        
        # If no model provided, return the cv object
        if model is None or X is None or y is None:
            return cv
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, 
            return_train_score=return_train_score,
            return_estimator=return_estimator
        )
        
        return cv_results
    
    @staticmethod
    def custom_cross_validation(X, y, model, cv, scoring=None, groups=None, 
                              return_estimator=False, return_train_score=True):
        """
        Perform custom cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        model : estimator object
            The model to evaluate
        cv : cross-validation generator or iterable
            Determines the cross-validation splitting strategy
        scoring : str, callable, or None, default=None
            Scoring metric to use
        groups : array-like, default=None
            Group labels for the samples
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, groups=groups, scoring=scoring, 
            return_train_score=return_train_score,
            return_estimator=return_estimator
        )
        
        return cv_results
    
    @staticmethod
    def multi_metric_cross_validation(X, y, model, cv=5, scoring=None, groups=None,
                                    return_estimator=False, return_train_score=True):
        """
        Perform cross-validation with multiple metrics.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        model : estimator object
            The model to evaluate
        cv : int, cross-validation generator or iterable, default=5
            Determines the cross-validation splitting strategy
        scoring : dict, list, or None, default=None
            Scoring metrics to use
        groups : array-like, default=None
            Group labels for the samples
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # If scoring is None, use default metrics
        if scoring is None:
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, groups=groups, scoring=scoring, 
            return_train_score=return_train_score,
            return_estimator=return_estimator
        )
        
        return cv_results
    
    @staticmethod
    def plot_cv_results(cv_results, title='Cross-Validation Results', figsize=(12, 6)):
        """Plot cross-validation results."""
        return plot_cv_results(cv_results, title, figsize)
    
    @staticmethod
    def visualize_cv_splits(X, y, cv, groups=None, figsize=(12, 6)):
        """Visualize cross-validation splits."""
        return visualize_cv_splits(X, y, cv, groups, figsize)
    