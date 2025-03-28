import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from ..config import random_state

class PCAApplication:
    """
    A class for applying PCA (Principal Component Analysis) to combine features.
    Includes functionality for:
    - Standardization
    - PCA transformation
    - Variance ratio analysis
    - Component visualization
    - Feature contribution analysis
    """
    
    def __init__(self, n_components=None, standardize=True):
        """
        Initialize PCA application.
        
        Parameters:
        -----------
        n_components : int or float, default=None
            Number of components to keep:
            - If int, number of components to keep
            - If float between 0 and 1, proportion of variance to keep
            - If None, keep all components
        standardize : bool, default=True
            Whether to standardize the features before applying PCA
        """
        self.n_components = n_components
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.is_fitted = False
        self.feature_names = None
        self.transformed_feature_names = None
        
    def fit(self, X, columns=None):
        """
        Fit PCA to the data.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Features to transform
        columns : list, default=None
            List of column names to use. If None and X is a DataFrame, 
            uses all columns
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Handle column selection
        if isinstance(X, pd.DataFrame):
            if columns is not None:
                X = X[columns]
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = columns if columns is not None else [
                f'Feature_{i}' for i in range(X.shape[1])
            ]
        
        # Standardize if requested
        if self.standardize:
            X = self.scaler.fit_transform(X)
        
        # Initialize and fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=random_state)
        self.pca.fit(X)
        
        # Create transformed feature names
        self.transformed_feature_names = [
            f'PC{i+1}' for i in range(self.pca.n_components_)
        ]
        
        self.is_fitted = True
        return self
    
    def transform(self, X, columns=None):
        """
        Apply PCA transformation to the data.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Features to transform
        columns : list, default=None
            List of column names to use. If None and X is a DataFrame,
            uses all columns
            
        Returns:
        --------
        pandas.DataFrame
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        # Handle column selection
        if isinstance(X, pd.DataFrame):
            if columns is not None:
                X = X[columns]
            X = X.values
        
        # Standardize if requested
        if self.standardize:
            X = self.scaler.transform(X)
        
        # Transform data
        X_transformed = self.pca.transform(X)
        
        # Return as DataFrame
        return pd.DataFrame(
            X_transformed,
            columns=self.transformed_feature_names
        )
    
    def fit_transform(self, X, columns=None):
        """
        Fit PCA to the data and apply transformation.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Features to transform
        columns : list, default=None
            List of column names to use. If None and X is a DataFrame,
            uses all columns
            
        Returns:
        --------
        pandas.DataFrame
            Transformed features
        """
        return self.fit(X, columns).transform(X, columns)
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with component names and their explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return pd.DataFrame({
            'Component': self.transformed_feature_names,
            'Explained Variance Ratio': self.pca.explained_variance_ratio_,
            'Cumulative Variance Ratio': np.cumsum(self.pca.explained_variance_ratio_)
        })
    
    def plot_explained_variance(self, figsize=(12, 6)):
        """
        Plot explained variance ratio and cumulative explained variance.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        variance_df = self.get_explained_variance_ratio()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot explained variance ratio
        sns.barplot(
            data=variance_df,
            x='Component',
            y='Explained Variance Ratio',
            ax=ax1
        )
        ax1.set_title('Explained Variance Ratio by Component')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot cumulative explained variance
        sns.lineplot(
            data=variance_df,
            x='Component',
            y='Cumulative Variance Ratio',
            marker='o',
            ax=ax2
        )
        ax2.set_title('Cumulative Explained Variance Ratio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
        ax2.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_component_loadings(self):
        """
        Get the loadings (coefficients) of original features on principal components.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature loadings for each component
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=self.transformed_feature_names,
            index=self.feature_names
        )
        
        return loadings
    
    def plot_component_loadings(self, n_components=None, figsize=(12, 8)):
        """
        Plot feature loadings heatmap.
        
        Parameters:
        -----------
        n_components : int, default=None
            Number of components to plot. If None, plots all components
        figsize : tuple, default=(12, 8)
            Figure size
        """
        loadings = self.get_component_loadings()
        
        if n_components is not None:
            loadings = loadings.iloc[:, :n_components]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            loadings,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('Feature Loadings on Principal Components')
        plt.tight_layout()
        plt.show()
    
    def plot_2d_projection(self, X, y=None, components=None):
        """
        Plot 2D projection of the data.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Features to project
        y : array-like, default=None
            Target variable for coloring
        components : list, default=None
            List of two component indices to plot. If None, uses first two components.
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        # Check if we have enough components for 2D projection
        if self.pca.n_components_ < 2:
            print("Warning: Cannot create 2D projection with less than 2 components.")
            return
        
        # Transform data if not already transformed
        if isinstance(X, pd.DataFrame) and X.columns[0] != self.transformed_feature_names[0]:
            X_transformed = self.transform(X)
        else:
            X_transformed = X
        
        # Set default components if not specified
        if components is None:
            components = [0, 1]
        
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            scatter = plt.scatter(
                X_transformed.iloc[:, components[0]],
                X_transformed.iloc[:, components[1]],
                c=y
            )
            plt.colorbar(scatter)
        else:
            plt.scatter(
                X_transformed.iloc[:, components[0]],
                X_transformed.iloc[:, components[1]]
            )
        
        plt.xlabel(f'{self.transformed_feature_names[components[0]]}')
        plt.ylabel(f'{self.transformed_feature_names[components[1]]}')
        plt.title('2D Projection of PCA Components')
        plt.tight_layout()
        plt.show()
    
    def get_optimal_components(self, variance_threshold=0.95):
        """
        Get the optimal number of components for a given variance threshold.
        
        Parameters:
        -----------
        variance_threshold : float, default=0.95
            Minimum proportion of variance to retain
            
        Returns:
        --------
        int
            Optimal number of components
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        return n_components
    
    def get_reconstruction_error(self, X, columns=None):
        """
        Calculate reconstruction error (mean squared error).
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Original features
        columns : list, default=None
            List of column names to use
            
        Returns:
        --------
        float
            Reconstruction error
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        # Handle column selection
        if isinstance(X, pd.DataFrame):
            if columns is not None:
                X = X[columns]
            X = X.values
        
        # Standardize if requested
        if self.standardize:
            X = self.scaler.transform(X)
        
        # Transform and inverse transform
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Calculate error
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        
        return reconstruction_error

    def pca_runner(self, data, target_col=None, variance_threshold=0.95):
        """
        Runner function to perform PCA analysis on a dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data containing features
        target_col : str, default=None
            Name of target column if exists
        variance_threshold : float, default=0.95
            Threshold for optimal number of components
            
        Returns:
        --------
        dict
            Dictionary containing:
            - transformed_data: PCA transformed features
            - pca_model: Fitted PCA model
            - scaler: Fitted StandardScaler (if standardize=True)
            - feature_names: Original feature names
            - component_names: PCA component names
            - variance_ratios: Explained variance ratios
            - optimal_components: Optimal number of components
            - reconstruction_error: Reconstruction error
            - fitted_models: Dictionary containing fitted PCA and scaler objects
        """
        # Separate features and target
        if target_col is not None and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
            X = data
            y = None
        
        # Store original feature names
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        # Transform data
        X_transformed = self.fit_transform(X)
        
        # Create component names
        n_components = X_transformed.shape[1]
        component_names = [f"PC{i+1}" for i in range(n_components)]
        
        # Convert transformed data to DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed, columns=component_names, index=X.index)
        
        # Get variance ratios
        variance_df = self.get_explained_variance_ratio()
        
        # Get optimal components based on variance threshold
        optimal_n = self.get_optimal_components(variance_threshold)
        
        # Print comparison if optimal_n is different from current n_components
        if optimal_n != n_components:
            print(f"\nNote: Current components ({n_components}) differ from optimal components ({optimal_n})")
            print(f"Consider reinitializing with n_components={optimal_n} for optimal variance retention")
        
        # Calculate reconstruction error
        error = self.get_reconstruction_error(X)
        
        # Create visualizations
        print("\n=== PCA Analysis Results ===")
        print(f"Original features: {X.shape[1]}")
        print(f"Components kept: {n_components}")
        print(f"Explained variance with current components: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"Optimal components for {variance_threshold*100}% variance: {optimal_n}")
        print(f"Reconstruction error: {error:.6f}")
        print("\nExplained Variance Ratios:")
        print(variance_df)
        
        # Plot results
        self.plot_explained_variance()
        self.plot_component_loadings()
        
        # 2D projection sadece 2 veya daha fazla bileşen varsa yapılacak
        if n_components >= 2:
            if y is not None:
                print("\nPlotting 2D projection with target variable...")
                self.plot_2d_projection(X, y)
            else:
                print("\nPlotting 2D projection...")
                self.plot_2d_projection(X)
        else:
            print("\nSkipping 2D projection: Not enough components")
        
        # Store fitted models for later use
        fitted_models = {
            'pca': self.pca,
            'scaler': self.scaler if self.standardize else None
        }
        
        # Return results
        results = {
            'transformed_data': X_transformed,
            'pca_model': self,
            'scaler': self.scaler if self.standardize else None,
            'feature_names': feature_names,
            'component_names': component_names,
            'variance_ratios': variance_df,
            'optimal_components': optimal_n,
            'reconstruction_error': error,
            'fitted_models': fitted_models,
            'current_n_components': n_components,
            'total_variance_explained': self.pca.explained_variance_ratio_.sum()
        }
        
        return results



