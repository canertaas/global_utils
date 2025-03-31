from ..config import random_state as default_random_state
from .validation import Validation


def validate_data(X, y, groups=None, validation_method='stratified_k_fold', validation_params=None):
    """
    Split data based on the chosen validation strategy
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    groups : array-like, default=None
        Group labels for the samples (used for group-based validation)
    validation_method : str, default='stratified_k_fold'
        Validation method to use
    validation_params : dict, default=None
        Parameters for the validation method
        
    Returns:
    --------
    X_train, X_val, y_train, y_val : arrays
        The split data
    """
    if validation_params is None:
        validation_params = {}
    
    test_size = validation_params.get('test_size', 0.2)
    random_state = validation_params.get('random_state', default_random_state)
    
    if validation_method == 'train_test_split':
        return Validation.train_test_split(
            X, y,
            test_size=test_size,
            stratify=validation_params.get('stratify', y),
            random_state=random_state
        )
    
    elif validation_method == 'time_series_split':
        return Validation.time_series_train_test_split(
            X, y,
            test_size=test_size
        )
        
    elif validation_method == 'group_split' and groups is not None:
        splits = Validation.group_train_test_split(
            X, y, groups,
            test_size=test_size,
            random_state=random_state
        )
        # Return only X_train, X_test, y_train, y_test from the result
        return splits[0], splits[1], splits[2], splits[3]
        
    else:
        # Default to stratified split
        return Validation.train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )


def get_cross_validator(validation_method='stratified_k_fold', validation_params=None, X=None, y=None, groups=None):
    """
    Get the appropriate cross-validator based on the validation method
    
    Parameters:
    -----------
    validation_method : str, default='stratified_k_fold'
        Validation method to use
    validation_params : dict, default=None
        Parameters for the validation method
    X : array-like, default=None
        Features
    y : array-like, default=None
        Target variable
    groups : array-like, default=None
        Group labels for samples (for group-based validation)
        
    Returns:
    --------
    cv : cross-validator
        The cross-validator object
    """
    if validation_params is None:
        validation_params = {}
    
    n_splits = validation_params.get('n_splits', 5)
    random_state = validation_params.get('random_state', default_random_state)
    
    if validation_method == 'k_fold':
        return Validation.k_fold_cross_validation(
            X=None, y=None, model=None,
            n_splits=n_splits,
            random_state=random_state
        )
        
    elif validation_method == 'stratified_k_fold':
        return Validation.stratified_k_fold_cross_validation(
            X=None, y=None, model=None,
            n_splits=n_splits,
            random_state=random_state
        )
        
    elif validation_method == 'group_k_fold' and groups is not None:
        return Validation.group_k_fold_cross_validation(
            X=None, y=None, groups=groups, model=None,
            n_splits=n_splits
        )
        
    elif validation_method == 'time_series':
        return Validation.time_series_cross_validation(
            X=None, y=None, model=None,
            n_splits=n_splits,
            test_size=validation_params.get('test_size', None),
            gap=validation_params.get('gap', 0)
        )
        
    elif validation_method == 'combinatorial_purged' and groups is not None:
        return Validation.combinatorial_purged_group_k_fold(
            X=X, y=y, groups=groups,
            n_splits=n_splits,
            group_gap=validation_params.get('group_gap', 0),
            embargo_pct=validation_params.get('embargo_pct', 0.0)
        )
        
    else:
        # Default to stratified k-fold
        return Validation.stratified_k_fold_cross_validation(
            X=None, y=None, model=None,
            n_splits=n_splits,
            random_state=random_state
        )


def perform_cross_validation(model, X, y, groups=None, validation_method='stratified_k_fold', validation_params=None, scoring=None):
    """
    Perform cross-validation using the chosen validation method
    
    Parameters:
    -----------
    model : estimator
        The model to evaluate
    X : array-like
        Features
    y : array-like
        Target variable
    groups : array-like, default=None
        Group labels for samples (for group-based methods)
    validation_method : str, default='stratified_k_fold'
        Validation method to use
    validation_params : dict, default=None
        Parameters for the validation method
    scoring : str or callable, default=None
        Scoring metric to use
        
    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """
    if validation_params is None:
        validation_params = {}
    
    n_splits = validation_params.get('n_splits', 5)
    random_state = validation_params.get('random_state', default_random_state)
    
    if validation_method == 'k_fold':
        return Validation.k_fold_cross_validation(
            X=X, y=y, model=model,
            n_splits=n_splits,
            random_state=random_state,
            scoring=scoring
        )
        
    elif validation_method == 'stratified_k_fold':
        return Validation.stratified_k_fold_cross_validation(
            X=X, y=y, model=model,
            n_splits=n_splits,
            random_state=random_state,
            scoring=scoring
        )
        
    elif validation_method == 'group_k_fold' and groups is not None:
        return Validation.group_k_fold_cross_validation(
            X=X, y=y, groups=groups, model=model,
            n_splits=n_splits,
            scoring=scoring
        )
        
    elif validation_method == 'time_series':
        return Validation.time_series_cross_validation(
            X=X, y=y, model=model,
            n_splits=n_splits,
            test_size=validation_params.get('test_size', None),
            gap=validation_params.get('gap', 0),
            scoring=scoring
        )
        
    elif validation_method == 'combinatorial_purged' and groups is not None:
        return Validation.combinatorial_purged_group_k_fold_cross_validation(
            X=X, y=y, groups=groups, model=model,
            n_splits=n_splits,
            group_gap=validation_params.get('group_gap', 0),
            embargo_pct=validation_params.get('embargo_pct', 0.0),
            scoring=scoring
        )
        
    else:
        # Default to stratified k-fold
        return Validation.stratified_k_fold_cross_validation(
            X=X, y=y, model=model,
            n_splits=n_splits,
            random_state=random_state,
            scoring=scoring
        )


class GetValidation:
    """
    Utility class to manage validation strategies for machine learning models.
    
    This class provides methods to:
    - Split data using various validation strategies
    - Get cross-validators for different validation methods
    - Perform cross-validation using different validation methods
    """
    
    @staticmethod
    def validate_data(X, y, groups=None, validation_method='stratified_k_fold', validation_params=None):
        """Split data based on the chosen validation strategy"""
        return validate_data(X, y, groups, validation_method, validation_params)
    
    @staticmethod
    def get_cross_validator(validation_method='stratified_k_fold', validation_params=None, X=None, y=None, groups=None):
        """Get the appropriate cross-validator based on the validation method"""
        return get_cross_validator(validation_method, validation_params, X, y, groups)
    
    @staticmethod
    def perform_cross_validation(model, X, y, groups=None, validation_method='stratified_k_fold', validation_params=None, scoring=None):
        """Perform cross-validation using the chosen validation method"""
        return perform_cross_validation(model, X, y, groups, validation_method, validation_params, scoring)
