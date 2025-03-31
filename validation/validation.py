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
    
    @staticmethod
    def combinatorial_purged_group_k_fold(X, y, groups, n_splits=5, group_gap=0, embargo_pct=0.0):
        """
        Generate indices for Combinatorial Purged Group K-Fold cross-validation.
        
        This cross-validation strategy:
        1. Ensures no data leakage between training and test sets
        2. Implements purging (removing samples before/after test set)
        3. Implements embargoing (removing samples from the end of training)
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        groups : array-like
            Group labels for samples
        n_splits : int, default=5
            Number of folds
        group_gap : int, default=0
            Number of groups to purge between train and test sets
        embargo_pct : float, default=0.0
            Percentage of samples to embargo from the end of training set
        
        Yields:
        -------
        train_idx : ndarray
            Training set indices for each split
        test_idx : ndarray
            Test set indices for each split
        """
        if groups is None:
            raise ValueError("Groups parameter must be provided")
        
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_splits > n_groups:
            raise ValueError(f"Cannot have more splits ({n_splits}) than groups ({n_groups})")
        
        # Calculate fold size
        fold_size = n_groups // n_splits
        
        # Generate test group indices for each fold
        indices = np.arange(n_groups)
        test_starts = indices[::fold_size][:n_splits]
        
        for fold_start in test_starts:
            # Define test groups
            fold_end = fold_start + fold_size
            test_groups = unique_groups[fold_start:fold_end]
            
            # Create test mask
            test_mask = np.isin(groups, test_groups)
            test_indices = np.where(test_mask)[0]
            
            # Define train groups (excluding purged and test groups)
            purged_groups = []
            for test_group in test_groups:
                purged_groups.extend(range(
                    max(0, test_group - group_gap),
                    min(n_groups, test_group + group_gap + 1)
                ))
            
            train_groups = [g for g in unique_groups if g not in purged_groups]
            train_mask = np.isin(groups, train_groups)
            
            # Apply embargo if requested
            if embargo_pct > 0:
                n_embargo = int(len(test_groups) * embargo_pct)
                embargo_groups = np.zeros(n_groups, dtype=bool)
                for test_group in test_groups:
                    embargo_groups[max(0, test_group - n_embargo):test_group] = True
                embargo_mask = embargo_groups[groups]
                train_mask &= ~embargo_mask
            
            train_indices = np.where(train_mask)[0]
            
            yield train_indices, test_indices
    
    @staticmethod
    def combinatorial_purged_group_k_fold_cross_validation(X, y, groups, model=None, n_splits=5, 
                                                          group_gap=0, embargo_pct=0.0, 
                                                          scoring=None, return_estimator=False, 
                                                          return_train_score=True):
        """
        Create a Combinatorial Purged Group K-Fold cross-validator or perform cross-validation.
        
        Parameters:
        -----------
        X : array-like or None
            Features. If None, returns a generator function for splits.
        y : array-like or None
            Target variable. If None, returns a generator function for splits.
        groups : array-like
            Group labels for samples
        model : estimator object, default=None
            The model to evaluate. If None, returns only the cross-validator.
        n_splits : int, default=5
            Number of folds
        group_gap : int, default=0
            Number of groups to purge between train and test sets
        embargo_pct : float, default=0.0
            Percentage of samples to embargo from the end of training set
        scoring : str, callable, or None, default=None
            Scoring metric to use
        return_estimator : bool, default=False
            Whether to return the fitted estimator objects
        return_train_score : bool, default=True
            Whether to return training scores
        
        Returns:
        --------
        If model is None:
            cv_generator : generator
                Generator yielding train and test indices
        Else:
            dict
                Cross-validation results containing:
                - test_score: Array of scores for each fold
                - mean_test_score: Mean score across folds
                - std_test_score: Standard deviation of scores
        """
        # Create CV splitter function
        cv = Validation.combinatorial_purged_group_k_fold(
            X, y, groups, 
            n_splits=n_splits, 
            group_gap=group_gap, 
            embargo_pct=embargo_pct
        )
        
        # If no model, return the cross-validator
        if model is None:
            return cv
        
        # Convert generator to list for cross_val_score
        cv_list = list(cv)
        
        if return_estimator or return_train_score:
            # If we need to return estimators or train scores, use cross_validate
            cv_results = cross_validate(
                model, X, y, cv=cv_list, scoring=scoring,
                return_estimator=return_estimator,
                return_train_score=return_train_score
            )
            return cv_results
        else:
            # Otherwise use cross_val_score which is simpler
            scores = cross_val_score(model, X, y, cv=cv_list, scoring=scoring)
            
            return {
                'test_score': scores,
                'mean_test_score': scores.mean(),
                'std_test_score': scores.std()
            }