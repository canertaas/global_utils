import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay


class ModelExplainer:
    """
    A comprehensive class for explaining machine learning models using various techniques
    including SHAP, feature importance, and partial dependence plots.
    """
    
    @staticmethod
    def explain_model(model, X, model_name, feature_names=None, output_dir=None, top_n_features=20, sample_size=10000):
        """
        Generate comprehensive model explanations and save visualizations
        
        Parameters:
        -----------
        model : trained model object
            The machine learning model to explain
        X : array-like or DataFrame
            Data for explanation
        model_name : str
            Name of the model
        feature_names : list, default=None
            Names of features (used if X is not a DataFrame)
        output_dir : str, default=None
            Directory to save outputs (if None, visualizations are displayed but not saved)
        top_n_features : int, default=10
            Number of top features to include in some visualizations
        sample_size : int, default=1000
            Maximum number of samples to use for SHAP computation to save memory
            
        Returns:
        --------
        dict
            Dictionary with explanation results
        """
        # Prepare output directory if provided
        if output_dir:
            # Clean model name for directory name (remove invalid characters)
            safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
            model_dir = output_dir
            os.makedirs(model_dir, exist_ok=True)
        else:
            model_dir = None
            safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
        
        # Get feature names from DataFrame or use provided names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        elif feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Prepare X with feature names for better visualizations
        X_with_names = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)
        
        # Results dictionary
        results = {}
        
        # Generate explanations
        try:
            # 1. Feature importance
            print(f"Calculating feature importance for {model_name}...")
            importance = ModelExplainer.get_feature_importance(model, X, feature_names)
            results['feature_importance'] = importance
            
            if model_dir:
                ModelExplainer.plot_feature_importance(
                    importance, feature_names, model_name, 
                    save_path=os.path.join(model_dir, f'{safe_model_name}_feature_importance.png')
                )
                
                # Save as CSV
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                importance_df.to_csv(os.path.join(model_dir, f'{safe_model_name}_feature_importance.csv'), index=False)
            
            # 2. SHAP analysis
            print(f"Generating SHAP values for {model_name}...")
            shap_values = ModelExplainer.generate_shap_values(
                model, X_with_names, model_name, sample_size=sample_size
            )
            results['shap_values'] = shap_values
            
            if model_dir and shap_values is not None:
                print(f"Creating SHAP visualizations for {model_name}...")
                ModelExplainer.create_shap_visualizations(
                    shap_values, X_with_names, model_name, 
                    save_dir=model_dir,
                    top_n_features=top_n_features,
                    safe_model_name=safe_model_name
                )
            
            # 3. Partial dependence plots for top features
            if hasattr(model, 'predict') and model_dir:
                print(f"Creating partial dependence plots for {model_name}...")
                # Get top features by importance
                top_features = np.argsort(importance)[-top_n_features:]
                top_feature_names = [feature_names[i] for i in top_features]
                
                ModelExplainer.create_partial_dependence_plots(
                    model, X_with_names, top_feature_names, model_name,
                    save_dir=model_dir,
                    safe_model_name=safe_model_name
                )
            
        except Exception as e:
            print(f"Error in explain_model for {model_name}: {str(e)}")
        
        return results
    
    @staticmethod
    def get_feature_importance(model, X, feature_names=None, n_repeats=10):
        """
        Get feature importance from the model.
        
        Parameters:
        -----------
        model : trained model object
            The machine learning model
        X : array-like
            Data used for permutation importance if needed
        feature_names : list, default=None
            Names of features
        n_repeats : int, default=10
            Number of times to permute each feature for permutation importance
            
        Returns:
        --------
        array
            Feature importance values
        """
        try:
            # Try direct feature importance from model
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            
            # For linear models with coefficients
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    return np.abs(model.coef_)
                else:
                    return np.mean(np.abs(model.coef_), axis=0)
            
            # Fall back to permutation importance
            else:
                from sklearn.inspection import permutation_importance
                # Use a subset of data for permutation importance if too large
                if hasattr(X, 'shape') and X.shape[0] > 5000:
                    sample_indices = np.random.choice(X.shape[0], 5000, replace=False)
                    X_sample = X[sample_indices] if not isinstance(X, pd.DataFrame) else X.iloc[sample_indices]
                else:
                    X_sample = X
                
                perm_importance = permutation_importance(
                    model, X_sample, y=None, 
                    n_repeats=n_repeats, 
                    random_state=42, 
                    n_jobs=-1
                )
                return perm_importance.importances_mean
                
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            # Return uniform importance if all else fails
            if hasattr(X, 'shape'):
                return np.ones(X.shape[1]) / X.shape[1]
            elif feature_names:
                return np.ones(len(feature_names)) / len(feature_names)
            else:
                return np.array([1.0])
    
    @staticmethod
    def plot_feature_importance(importance, feature_names, model_name, top_n=20, figsize=(12, 10), save_path=None):
        """
        Plot feature importance as a horizontal bar chart.
        
        Parameters:
        -----------
        importance : array-like
            Feature importance values
        feature_names : list
            Names of features
        model_name : str
            Name of the model for the title
        top_n : int, default=20
            Number of top features to include (None for all)
        figsize : tuple, default=(12, 10)
            Figure size
        save_path : str, default=None
            Path to save the figure
        """
        try:
            # Sort features by importance
            indices = np.argsort(importance)
            
            # Take top N if specified
            if top_n is not None and top_n < len(indices):
                indices = indices[-top_n:]
            
            sorted_names = [feature_names[i] for i in indices]
            sorted_importance = importance[indices]
            
            # Create plot
            plt.figure(figsize=figsize)
            plt.barh(range(len(sorted_names)), sorted_importance, align='center', color='steelblue')
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'Feature Importance for {model_name}', fontsize=14, fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error plotting feature importance for {model_name}: {str(e)}")
    
    @staticmethod
    def generate_shap_values(model, X, model_name, sample_size=1000):
        """
        Generate SHAP values for the model.
        
        Parameters:
        -----------
        model : trained model object
            The machine learning model
        X : DataFrame or array-like
            Data for SHAP explanation
        model_name : str
            Name of the model
        sample_size : int, default=1000
            Maximum number of samples to use for SHAP computation
            
        Returns:
        --------
        object
            SHAP values object
        """
        try:
            # Sample data if it's too large
            if X.shape[0] > sample_size:
                sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
                X_sample = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
            else:
                X_sample = X
            
            # Determine the appropriate SHAP explainer based on model type
            model_type = model_name.lower()
            
            # For tree-based models
            if any(m in model_type for m in ['xgboost', 'xgb', 'lightgbm', 'lgbm', 'catboost', 'randomforest', 'random_forest', 'rf', 'gradient']):
                explainer = shap.TreeExplainer(model)
            
            # For linear models
            elif any(m in model_type for m in ['linear', 'logistic', 'regression', 'svm', 'svc']):
                explainer = shap.LinearExplainer(model, X_sample)
            
            # For deep learning models (if applicable)
            elif any(m in model_type for m in ['neural', 'deep', 'nn', 'network']):
                background = X_sample.iloc[:100] if hasattr(X_sample, 'iloc') else X_sample[:100]
                explainer = shap.DeepExplainer(model, background)
            
            # For all other models
            else:
                # Use KernelExplainer with a smaller subset for efficiency
                subset_size = min(100, X_sample.shape[0])
                background = X_sample.iloc[:subset_size] if hasattr(X_sample, 'iloc') else X_sample[:subset_size]
                pred_func = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                explainer = shap.KernelExplainer(pred_func, background)
            
            # Calculate SHAP values
            shap_values = explainer(X_sample)
            return shap_values
            
        except Exception as e:
            print(f"Error generating SHAP values for {model_name}: {str(e)}")
            return None
    
    @staticmethod
    def create_shap_visualizations(shap_values, X, model_name, save_dir=None, top_n_features=10, safe_model_name=None):
        """
        Create and save various SHAP visualizations.
        
        Parameters:
        -----------
        shap_values : object
            SHAP values from a SHAP explainer
        X : DataFrame
            Data used for explanation
        model_name : str
            Name of the model
        save_dir : str, default=None
            Directory to save visualizations
        top_n_features : int, default=10
            Number of top features to include in some visualizations
        safe_model_name : str, default=None
            Safe version of model name for file naming
        """
        if safe_model_name is None:
            safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
            
        try:
            # 1. Summary plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values, X, show=False)
            plt.title(f'SHAP Summary for {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{safe_model_name}_shap_summary.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # 2. Bar summary plot (feature importance)
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance for {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{safe_model_name}_shap_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # 3. Dependence plots for top features
            try:
                # Get the feature importance based on SHAP values
                feature_importance = np.abs(shap_values.values).mean(0)
                top_indices = feature_importance.argsort()[-top_n_features:]
                
                for idx in top_indices:
                    plt.figure(figsize=(12, 8))
                    feature_name = X.columns[idx]
                    # Clean feature name for file naming
                    safe_feature_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in feature_name)
                    
                    shap.dependence_plot(idx, shap_values.values, X, show=False)
                    plt.title(f'SHAP Dependence Plot for {feature_name} ({model_name})', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    if save_dir:
                        plt.savefig(
                            os.path.join(save_dir, f'{safe_model_name}_shap_dependence_{safe_feature_name}.png'), 
                            dpi=300, bbox_inches='tight'
                        )
                        plt.close()
                    else:
                        plt.show()
            except Exception as e:
                print(f"Error creating SHAP dependence plots for {model_name}: {str(e)}")
            
            # 4. SHAP waterfall plot for a typical instance
            try:
                # Use a middle instance to show a typical example
                middle_idx = len(X) // 2
                plt.figure(figsize=(14, 10))
                
                # Ensure that shap_values is not a matrix
                if shap_values.ndim > 1:
                    shap.plots.waterfall(shap_values[middle_idx, :], show=False)  # Select a single row
                else:
                    shap.plots.waterfall(shap_values[middle_idx], show=False)  # For single output models
                
                plt.title(f'SHAP Waterfall Plot for {model_name} (Sample Instance)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{safe_model_name}_shap_waterfall.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
            except Exception as e:
                print(f"Error creating SHAP waterfall plot for {model_name}: {str(e)}")
                
        except Exception as e:
            print(f"Error in create_shap_visualizations for {model_name}: {str(e)}")
    
    @staticmethod
    def create_partial_dependence_plots(model, X, features, model_name, save_dir=None, grid_resolution=50, safe_model_name=None):
        """
        Create and save partial dependence plots.
        
        Parameters:
        -----------
        model : trained model object
            The machine learning model
        X : DataFrame
            Data used for explanation
        features : list
            List of feature names or indices to plot
        model_name : str
            Name of the model
        save_dir : str, default=None
            Directory to save visualizations
        grid_resolution : int, default=50
            Resolution of the grid for partial dependence calculation
        safe_model_name : str, default=None
            Safe version of model name for file naming
        """
        if safe_model_name is None:
            safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
            
        try:
            # Ensure save directory exists if provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            # Convert feature names to indices if needed
            if isinstance(features[0], str):
                feature_indices = [list(X.columns).index(feat) for feat in features]
            else:
                feature_indices = features
                features = [X.columns[i] for i in feature_indices]
            
            # Limit data size for partial dependence calculation
            X_sample = X
            if X.shape[0] > 5000:
                sample_indices = np.random.choice(X.shape[0], 5000, replace=False)
                X_sample = X.iloc[sample_indices]
                
            # Create figure for all features
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Generate partial dependence plots
            PartialDependenceDisplay.from_estimator(
                model, X_sample, feature_indices, 
                kind="average", subsample=1000,
                n_jobs=-1, grid_resolution=grid_resolution, 
                random_state=42, ax=ax
            )
            
            # Customize appearance
            fig.suptitle(f'Partial Dependence Plots for {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save or show
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{safe_model_name}_partial_dependence.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
            # Also create individual plots for each feature
            if save_dir:
                for i, feat in enumerate(features):
                    # Clean feature name for file naming
                    safe_feature_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in feat)
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    PartialDependenceDisplay.from_estimator(
                        model, X_sample, [feature_indices[i]], 
                        kind="average", subsample=1000,
                        n_jobs=-1, grid_resolution=grid_resolution, 
                        random_state=42, ax=ax
                    )
                    fig.suptitle(f'Partial Dependence for {feat} ({model_name})', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.92)
                    plt.savefig(
                        os.path.join(save_dir, f'{safe_model_name}_partial_dependence_{safe_feature_name}.png'), 
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close()
                
        except Exception as e:
            print(f"Error creating partial dependence plots for {model_name}: {str(e)}")
