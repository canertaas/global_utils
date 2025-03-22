import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, GroupKFold, 
                                     TimeSeriesSplit, cross_validate)

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
    
    It also provides visualization tools for cross-validation results.
    """
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
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
    def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=None, stratify=None):
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
    def group_train_test_split(X, y, groups, test_size=0.2, random_state=None):
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
    def k_fold_cross_validation(X, y, model, n_splits=5, random_state=None, scoring='accuracy'):
        """
        Perform K-Fold cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        model : estimator object
            The model to evaluate
        n_splits : int, default=5
            Number of folds
        random_state : int, default=None
            Controls the randomness of the CV splitter
        scoring : str or callable, default='accuracy'
            Scoring metric to use
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Create KFold object
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=kf, scoring=scoring, 
            return_train_score=True, return_estimator=True
        )
        
        return cv_results
    
    @staticmethod
    def stratified_k_fold_cross_validation(X, y, model, n_splits=5, random_state=None, scoring='accuracy'):
        """
        Perform Stratified K-Fold cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        model : estimator object
            The model to evaluate
        n_splits : int, default=5
            Number of folds
        random_state : int, default=None
            Controls the randomness of the CV splitter
        scoring : str or callable, default='accuracy'
            Scoring metric to use
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Create StratifiedKFold object
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=skf, scoring=scoring, 
            return_train_score=True, return_estimator=True
        )
        
        return cv_results
    
    @staticmethod
    def group_k_fold_cross_validation(X, y, groups, model, n_splits=5, scoring='accuracy'):
        """
        Perform Group K-Fold cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        groups : array-like
            Group labels for the samples
        model : estimator object
            The model to evaluate
        n_splits : int, default=5
            Number of folds
        scoring : str or callable, default='accuracy'
            Scoring metric to use
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Create GroupKFold object
        gkf = GroupKFold(n_splits=n_splits)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=gkf, groups=groups, scoring=scoring, 
            return_train_score=True, return_estimator=True
        )
        
        return cv_results
    
    @staticmethod
    def time_series_cross_validation(X, y, model, n_splits=5, test_size=None, gap=0, scoring='accuracy'):
        """
        Perform Time Series cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        model : estimator object
            The model to evaluate
        n_splits : int, default=5
            Number of splits
        test_size : int, default=None
            Number of samples in each test set
        gap : int, default=0
            Number of samples to exclude from the end of each train set before the test set
        scoring : str or callable, default='accuracy'
            Scoring metric to use
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Create TimeSeriesSplit object
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=tscv, scoring=scoring, 
            return_train_score=True, return_estimator=True
        )
        
        return cv_results
    
    @staticmethod
    def custom_cross_validation(X, y, model, cv, scoring='accuracy', groups=None):
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
        scoring : str or callable, default='accuracy'
            Scoring metric to use
        groups : array-like, default=None
            Group labels for the samples
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, groups=groups, scoring=scoring, 
            return_train_score=True, return_estimator=True
        )
        
        return cv_results
    
    @staticmethod
    def multi_metric_cross_validation(X, y, model, cv=5, scoring=None, groups=None):
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
            return_train_score=True, return_estimator=True
        )
        
        return cv_results
    
    @staticmethod
    def plot_cv_results(cv_results, title='Cross-Validation Results', figsize=(12, 6)):
        """
        Plot cross-validation results.
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results from cross_validate
        title : str, default='Cross-Validation Results'
            Plot title
        figsize : tuple, default=(12, 6)
            Figure size
        """
        # Extract metrics
        metrics = [key.replace('test_', '') for key in cv_results.keys() 
                  if key.startswith('test_') and key != 'test_score']
        
        if not metrics:
            metrics = ['score']
        
        # Create figure
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            train_scores = cv_results.get(f'train_{metric}', cv_results.get('train_score', []))
            test_scores = cv_results.get(f'test_{metric}', cv_results.get('test_score', []))
            
            # Calculate mean and std
            train_mean = np.mean(train_scores)
            train_std = np.std(train_scores)
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores)
            
            # Plot
            axes[i].bar([0, 1], [train_mean, test_mean], yerr=[train_std, test_std], 
                       capsize=10, color=['blue', 'orange'])
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['Train', 'Test'])
            axes[i].set_ylabel(f'{metric.capitalize()} Score')
            axes[i].set_title(f'{metric.capitalize()} (Mean ± Std)')
            
            # Add text
            axes[i].text(0, train_mean, f'{train_mean:.4f}±{train_std:.4f}', 
                        ha='center', va='bottom')
            axes[i].text(1, test_mean, f'{test_mean:.4f}±{test_std:.4f}', 
                        ha='center', va='bottom')
        
        # Set title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    @staticmethod
    def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5),
                           scoring='accuracy', n_jobs=None, figsize=(10, 6)):
        """
        Plot learning curve.
        
        Parameters:
        -----------
        estimator : estimator object
            The model to evaluate
        X : array-like
            Features
        y : array-like
            Target variable
        cv : int, cross-validation generator or iterable, default=5
            Determines the cross-validation splitting strategy
        train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
            Relative or absolute numbers of training examples
        scoring : str or callable, default='accuracy'
            Scoring metric to use
        n_jobs : int, default=None
            Number of jobs to run in parallel
        figsize : tuple, default=(10, 6)
            Figure size
        """
        from sklearn.model_selection import learning_curve
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=train_sizes, scoring=scoring
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.title(f'Learning Curve ({scoring})')
        plt.xlabel('Training examples')
        plt.ylabel(f'{scoring.capitalize()} score')
        plt.grid()
        
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1, color='orange')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        
        plt.legend(loc='best')
        plt.show()
    
    @staticmethod
    def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5,
                             scoring='accuracy', n_jobs=None, figsize=(10, 6)):
        """
        Plot validation curve.
        
        Parameters:
        -----------
        estimator : estimator object
            The model to evaluate
        X : array-like
            Features
        y : array-like
            Target variable
        param_name : str
            Name of the parameter to vary
        param_range : array-like
            Range of values for the parameter
        cv : int, cross-validation generator or iterable, default=5
            Determines the cross-validation splitting strategy
        scoring : str or callable, default='accuracy'
            Scoring metric to use
        n_jobs : int, default=None
            Number of jobs to run in parallel
        figsize : tuple, default=(10, 6)
            Figure size
        """
        from sklearn.model_selection import validation_curve
        
        # Calculate validation curve
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.title(f'Validation Curve ({scoring})')
        plt.xlabel(param_name)
        plt.ylabel(f'{scoring.capitalize()} score')
        plt.grid()
        
        plt.fill_between(param_range, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(param_range, test_mean - test_std,
                         test_mean + test_std, alpha=0.1, color='orange')
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.plot(param_range, test_mean, 'o-', color='orange', label='Cross-validation score')
        
        plt.legend(loc='best')
        plt.show()
    
    @staticmethod
    def visualize_cv_splits(X, y, cv, groups=None, figsize=(12, 6)):
        """
        Visualize cross-validation splits.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target variable
        cv : cross-validation generator
            The cross-validation splitter to visualize
        groups : array-like, default=None
            Group labels for the samples
        figsize : tuple, default=(12, 6)
            Figure size
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Get splits
        n_splits = getattr(cv, 'n_splits', 5)
        splits = list(cv.split(X, y, groups))
        
        # Create matrix to visualize splits
        n_samples = len(X)
        matrix = np.zeros((n_splits, n_samples))
        
        # Fill matrix
        for i, (train_idx, test_idx) in enumerate(splits):
            matrix[i, train_idx] = 1  # Training samples
            matrix[i, test_idx] = 2   # Test samples
        
        # Plot
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.viridis)
        plt.yticks(np.arange(n_splits), [f'Split {i+1}' for i in range(n_splits)])
        plt.xlabel('Sample index')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plt.cm.viridis(0.33), label='Training set'),
            Patch(facecolor=plt.cm.viridis(0.66), label='Test set')
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=2)
        
        plt.title('Cross-validation splits')
        plt.tight_layout()
        plt.show()