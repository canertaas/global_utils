import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from ..config import random_state


class Sampler(BaseEstimator, TransformerMixin):
    """
    Sampler class applies various sampling methods to address class imbalance in datasets
    and increase data size.
    
    Parameters:
    -------------
    method: str, default='smote'
        Sampling method to use: 'smote', 'random_oversampler', 'random_undersampler'
    
    scale_factor: float, default=1.0
        Factor to scale the dataset size. For example:
        - scale_factor=2.0: Double the dataset size
        - scale_factor=0.5: Reduce the dataset size by half
    
    class_ratio: float, dict or None, default=None
        Target ratio for classes:
        - If float: Represents the ratio of minority to majority class (for binary classification)
        - If dict: Specific proportion for each class, e.g., {0: 0.7, 1: 0.3}
        - If None: Maintains original class distribution
    
    random_state: int, default=None
        Seed for random number generator.
    
    k_neighbors: int, default=5
        Number of k nearest neighbors for SMOTE.
    """
    
    def __init__(self, method='smote', scale_factor=1.0, class_ratio=None, 
                 random_state=random_state, k_neighbors=5):
        self.method = method.lower()
        self.scale_factor = scale_factor
        self.class_ratio = class_ratio
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampler = None
        
    def _calculate_sampling_strategy(self, y):
        """
        Calculates the sampling strategy based on scale_factor and class_ratio.
        
        Parameters:
        -----------
        y: array-like
            Target labels
            
        Returns:
        --------
        dict: Sampling strategy dictionary with target counts for each class
        """
        # Count original class distribution
        unique, counts = np.unique(y, return_counts=True)
        original_counts = dict(zip(unique, counts))
        total_samples = len(y)
        
        # Calculate target total samples based on scale_factor
        target_total = int(total_samples * self.scale_factor)
        
        # Calculate class proportions based on class_ratio
        if self.class_ratio is None:
            # Maintain original class distribution
            class_proportions = {cls: count / total_samples for cls, count in original_counts.items()}
        elif isinstance(self.class_ratio, dict):
            # Use specified class ratio dictionary
            class_proportions = self.class_ratio.copy()
            
            # Normalize if needed
            sum_proportions = sum(class_proportions.values())
            if sum_proportions != 1.0:
                class_proportions = {cls: prop / sum_proportions for cls, prop in class_proportions.items()}
        elif isinstance(self.class_ratio, (int, float)):
            # For binary classification with a single ratio value
            if len(original_counts) != 2:
                raise ValueError("Float class_ratio can only be used with binary classification problems")
            
            # Identify minority and majority classes
            minority_class = min(original_counts, key=original_counts.get)
            majority_class = max(original_counts, key=original_counts.get)
            
            # Calculate proportions based on the ratio
            # ratio = minority / majority
            ratio_sum = 1.0 + self.class_ratio
            minority_prop = self.class_ratio / ratio_sum
            majority_prop = 1.0 / ratio_sum
            
            class_proportions = {
                minority_class: minority_prop,
                majority_class: majority_prop
            }
        else:
            raise ValueError("class_ratio must be None, a float, or a dictionary")
        
        # Calculate target counts
        target_counts = {cls: int(target_total * prop) for cls, prop in class_proportions.items()}
        
        # Ensure all classes have at least one sample
        for cls in target_counts:
            target_counts[cls] = max(1, target_counts[cls])
            
        return target_counts
    
    def _initialize_sampler(self, sampling_strategy):
        """
        Initializes the sampler based on the specified method.
        
        Parameters:
        -----------
        sampling_strategy: dict
            Target number of samples for each class
        """
        # Ensure k_neighbors is valid for SMOTE (must be less than min class size)
        k_neighbors = self.k_neighbors
        if self.method == 'smote':
            min_samples = min(sampling_strategy.values())
            k_neighbors = min(k_neighbors, min_samples - 1) if min_samples > 1 else 1
        
        if self.method == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k_neighbors
            )
        elif self.method == 'random_oversampler':
            self.sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'random_undersampler':
            self.sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Invalid method: {self.method}. Please use 'smote', 'random_oversampler' "
                f"or 'random_undersampler'."
            )
    
    def fit_resample(self, X, y):
        """
        Resamples the data according to scale_factor and class_ratio.
        
        Parameters:
        -------------
        X: array-like, shape (n_samples, n_features)
            Feature vectors.
        
        y: array-like, shape (n_samples,)
            Target values.
            
        Returns:
        ------
        X_resampled: array-like, shape (n_samples_new, n_features)
            Resampled feature vectors.
            
        y_resampled: array-like, shape (n_samples_new)
            Resampled target values.
        """
        # Calculate sampling strategy
        sampling_strategy = self._calculate_sampling_strategy(y)
        
        # Initialize sampler with calculated strategy
        self._initialize_sampler(sampling_strategy)
        
        # Print information about resampling
        print(f"Original dataset size: {len(y)} samples")
        print(f"Target dataset size: {sum(sampling_strategy.values())} samples (scale factor: {self.scale_factor})")
        print(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"Target class distribution: {sampling_strategy}")
        
        # Perform resampling
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        # Print results
        print(f"Resampled dataset size: {len(y_resampled)} samples")
        print(f"Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        return X_resampled, y_resampled
    
    def fit(self, X, y=None):
        """Implements fit method for Transformer API compatibility."""
        return self
    
    def transform(self, X, y=None):
        """
        Implements transform method for Transformer API compatibility.
        Note: This method only returns X, it does not perform sampling.
        Use fit_resample method for sampling.
        """
        return X
    
    def fit_transform(self, X, y=None):
        """
        If y is provided, resamples X and y.
        Otherwise, just returns X.
        """
        if y is not None:
            X_resampled, y_resampled = self.fit_resample(X, y)
            return X_resampled
        return X
    
    def run(self, X, y):
        """
        Convenience method to run the sampling process in one step.
        
        Parameters:
        -------------
        X: array-like, shape (n_samples, n_features)
            Feature vectors.
        
        y: array-like, shape (n_samples,)
            Target values.
            
        Returns:
        ------
        X_resampled: array-like, shape (n_samples_new, n_features)
            Resampled feature vectors.
            
        y_resampled: array-like, shape (n_samples_new)
            Resampled target values.
        """
        return self.fit_resample(X, y)


def get_class_distribution(y):
    """
    Calculates and prints class distribution.
    
    Parameters:
    -------------
    y: array-like
        Class labels.
    
    Returns:
    ------
    class_counts: dict
        Number of samples for each class.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print("Class Distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(y)*100:.2f}%)")
    
    return class_counts


def scale_dataset(X, y, scale_factor=4.0, class_ratio=None, method='smote', random_state=None):
    """
    Utility function to scale a dataset by a given factor while maintaining or adjusting class distribution.
    
    Parameters:
    -----------
    X: array-like
        Feature vectors
    
    y: array-like
        Target labels
    
    scale_factor: float, default=4.0
        Factor to scale the dataset size
    
    class_ratio: float, dict or None, default=None
        Target ratio for classes:
        - If float: Represents the ratio of minority to majority class
        - If dict: Specific proportion for each class
        - If None: Maintains original class distribution
    
    method: str, default='smote'
        Sampling method to use
    
    random_state: int, default=None
        Random seed for reproducibility
    
    Returns:
    --------
    X_scaled: array-like
        Scaled feature vectors
    
    y_scaled: array-like
        Scaled target labels
    """
    sampler = Sampler(
        method=method,
        scale_factor=scale_factor,
        class_ratio=class_ratio,
        random_state=random_state
    )
    
    return sampler.fit_resample(X, y)