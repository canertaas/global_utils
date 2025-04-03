import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
import warnings
from ..config import random_state
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """
    A comprehensive class for analyzing datasets, providing insights on:
    - Basic statistics
    - Missing values
    - Outliers
    - Distributions and normality
    - Correlations
    - Feature importance
    - Data visualization
    """
    
    def __init__(self, df, target_col=None):
        """
        Initialize the DatasetAnalyzer with a dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset to analyze
        target_col : str, optional
            The name of the target column for supervised learning tasks
        """
        self.df = df
        self.target_col = target_col
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if target_col and target_col in self.numerical_cols:
            self.numerical_cols.remove(target_col)
        if target_col and target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)
            
    def basic_info(self):
        """Display basic information about the dataset."""
        print(f"Dataset Shape: {self.df.shape}")
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nBasic Information:")
        self.df.info()
        
        print("\nSummary Statistics:")
        print(self.df.describe(include='all').T)
        
        # Class distribution for classification tasks
        if self.target_col and self.df[self.target_col].nunique() < 20:
            print(f"\nTarget Distribution ({self.target_col}):")
            target_counts = self.df[self.target_col].value_counts(normalize=True) * 100
            print(target_counts)
            
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.target_col, data=self.df)
            plt.title(f'Distribution of {self.target_col}')
            plt.xticks(rotation=45)
            plt.show()
    
    def missing_values_analysis(self):
        """Analyze missing values in the dataset."""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        missing_data = pd.DataFrame({
            'Missing Values': missing,
            'Percentage (%)': missing_percent
        })
        
        missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        
        print("Missing Values Analysis:")
        if len(missing_data) > 0:
            print(missing_data)
            
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            msno.matrix(self.df)
            plt.title('Missing Values Matrix')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            msno.bar(self.df)
            plt.title('Missing Values Bar Chart')
            plt.show()
            
            if len(missing_data) > 1:
                plt.figure(figsize=(12, 6))
                msno.heatmap(self.df)
                plt.title('Missing Values Correlation Heatmap')
                plt.show()
        else:
            print("No missing values found in the dataset.")
    
    def outlier_analysis(self, method='iqr', threshold=1.5):
        """
        Detect and visualize outliers in numerical columns.
        
        Parameters:
        -----------
        method : str, default='iqr'
            Method to detect outliers ('iqr' or 'zscore')
        threshold : float, default=1.5
            Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        """
        if not self.numerical_cols:
            print("No numerical columns found for outlier analysis.")
            return
        
        outlier_counts = {}
        
        for col in self.numerical_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = self.df[col][z_scores > threshold]
            
            outlier_counts[col] = len(outliers)
        
        outlier_df = pd.DataFrame({
            'Column': outlier_counts.keys(),
            'Outlier Count': outlier_counts.values(),
            'Percentage (%)': [count/len(self.df)*100 for count in outlier_counts.values()]
        }).sort_values('Outlier Count', ascending=False)
        
        print(f"Outlier Analysis (using {method.upper()}):")
        print(outlier_df)
        
        # Visualize outliers with boxplots
        n_cols = min(3, len(self.numerical_cols))
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 5))
        for i, col in enumerate(self.numerical_cols):
            plt.subplot(n_rows, n_cols, i+1)
            
            # If target column exists and has few unique values, separate boxplots by target
            if self.target_col and self.df[self.target_col].nunique() <= 10:
                sns.boxplot(x=self.target_col, y=col, data=self.df)
                plt.title(f'Boxplot of {col} by {self.target_col}')
                plt.xticks(rotation=45)
            else:
                sns.boxplot(y=self.df[col])
                plt.title(f'Boxplot of {col}')
                
            plt.tight_layout()
        plt.show()
    
    def distribution_analysis(self):
        """Analyze the distribution of numerical features and test for normality."""
        if not self.numerical_cols:
            print("No numerical columns found for distribution analysis.")
            return
        
        normality_results = []
        
        n_cols = min(3, len(self.numerical_cols))
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 5))
        for i, col in enumerate(self.numerical_cols):
            # Shapiro-Wilk test for normality
            sample = self.df[col].dropna()
            # Limit sample size for Shapiro-Wilk test
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=random_state)
            
            stat, p_value = stats.shapiro(sample)
            is_normal = p_value > 0.05
            
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurt()
            
            normality_results.append({
                'Column': col,
                'Shapiro p-value': p_value,
                'Normal Distribution': is_normal,
                'Skewness': skewness,
                'Kurtosis': kurtosis
            })
            
            # Plot histogram with KDE
            plt.subplot(n_rows, n_cols, i+1)
            
            # If target column exists and has few unique values, separate histograms by target
            if self.target_col and self.df[self.target_col].nunique() <= 5:
                for target_value in self.df[self.target_col].unique():
                    subset = self.df[self.df[self.target_col] == target_value]
                    sns.histplot(subset[col].dropna(), kde=True, 
                                label=f'{self.target_col}={target_value}', alpha=0.5)
                plt.legend()
                plt.title(f'Distribution of {col} by {self.target_col}')
            else:
                sns.histplot(self.df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                
            plt.tight_layout()
        plt.show()
        
        # QQ plots for normality check
        plt.figure(figsize=(15, n_rows * 5))
        for i, col in enumerate(self.numerical_cols):
            plt.subplot(n_rows, n_cols, i+1)
            stats.probplot(self.df[col].dropna(), dist="norm", plot=plt)
            plt.title(f'Q-Q Plot of {col}')
            plt.tight_layout()
        plt.show()
        
        normality_df = pd.DataFrame(normality_results)
        print("Normality Analysis:")
        print(normality_df)
    
    def correlation_analysis(self):
        """Analyze correlations between features."""
        if len(self.numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis.")
            return
        
        # Correlation matrix
        # Include target column in correlation analysis if it's numerical
        columns_for_corr = self.numerical_cols.copy()
        if self.target_col and self.target_col not in columns_for_corr:
            if self.df[self.target_col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(self.df[self.target_col]):
                columns_for_corr.append(self.target_col)
                print(f"Including target column '{self.target_col}' in correlation analysis.")
        
        corr_matrix = self.df[columns_for_corr].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # If target column exists and is numeric, show correlations with target
        if self.target_col and self.target_col in columns_for_corr:
            target_corrs = corr_matrix[self.target_col].drop(self.target_col).sort_values(ascending=False)
            
            print(f"\nFeature Correlations with Target ({self.target_col}):")
            print(target_corrs)
            
            # Visualize correlations with target
            plt.figure(figsize=(12, 8))
            sns.barplot(x=target_corrs.values, y=target_corrs.index)
            plt.title(f'Feature Correlations with {self.target_col}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.show()
        
        # Find highly correlated features
        high_corr_threshold = 0.7
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print(f"Highly Correlated Features (|r| >= {high_corr_threshold}):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.sort_values('Correlation', key=abs, ascending=False))
        else:
            print(f"No feature pairs with correlation >= {high_corr_threshold} found.")
    
    def categorical_analysis(self):
        """Analyze categorical features."""
        if not self.categorical_cols:
            print("No categorical columns found for analysis.")
            return
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            unique_count = len(value_counts)
            
            print(f"\nAnalysis of {col} (Unique Values: {unique_count})")
            
            if unique_count <= 20:  # Only show full distribution for columns with few categories
                print(value_counts)
                
                # Visualize distribution
                plt.figure(figsize=(12, 6))
                sns.countplot(y=col, data=self.df, order=value_counts.index)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.show()
                
                # If target column exists, show relationship with categorical feature
                if self.target_col and unique_count <= 10:
                    if self.df[self.target_col].nunique() <= 10:  # Classification
                        plt.figure(figsize=(12, 6))
                        sns.countplot(x=col, hue=self.target_col, data=self.df)
                        plt.title(f'Distribution of {col} by {self.target_col}')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.show()
            else:
                print("Top 10 categories:")
                print(value_counts.head(10))
                print(f"(Showing 10/{unique_count} categories)")
    
    def feature_importance(self):
        """Calculate feature importance if target column is provided."""
        if not self.target_col:
            print("No target column provided for feature importance analysis.")
            return
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            
            # Handle categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            print("\nFeature Importance Analysis using Multiple Models:")
            
            # Dictionary to store importance from all models
            all_importance = {}
            
            # 1. Tree-based model importance
            print("\n1. Tree-based Model Importance")
            if self.df[self.target_col].nunique() < 20:  # Classification
                tree_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                model_type = "classification"
            else:  # Regression
                tree_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                model_type = "regression"
            
            # Fit tree-based model
            tree_model.fit(X, y)
            
            # Get feature importance
            tree_importance = tree_model.feature_importances_
            
            # Create DataFrame for visualization
            tree_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': tree_importance
            }).sort_values('Importance', ascending=False)
            
            print("Top 20 features (Tree-based model):")
            print(tree_importance_df.head(20))
            
            # Store in all_importance
            all_importance['Tree-based'] = {feature: importance for feature, importance in zip(X.columns, tree_importance)}
            
            # 2. Linear model importance
            print("\n2. Linear Model Importance")
            
            # Scale features for linear model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Choose appropriate linear model
            if model_type == "classification":
                if self.df[self.target_col].nunique() == 2:  # Binary classification
                    from sklearn.linear_model import LogisticRegression
                    linear_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=random_state)
                    print("Using Logistic Regression (binary classification)")
                else:  # Multi-class classification
                    from sklearn.linear_model import LogisticRegression
                    linear_model = LogisticRegression(penalty='l2', C=1.0, solver='saga', multi_class='multinomial', random_state=random_state)
                    print("Using Logistic Regression (multi-class classification)")
            else:  # Regression
                linear_model = Ridge(alpha=1.0, random_state=random_state)
                print("Using Ridge Regression")
            
            try:
                # Fit linear model
                linear_model.fit(X_scaled, y)
                
                # Get feature importance (coefficients)
                if hasattr(linear_model, 'coef_'):
                    if len(linear_model.coef_.shape) > 1:  # Multi-class
                        # Average absolute coefficient across classes
                        linear_importance = np.mean(np.abs(linear_model.coef_), axis=0)
                    else:  # Binary class or regression
                        linear_importance = np.abs(linear_model.coef_)
                else:
                    raise AttributeError("Model does not have coef_ attribute")
                
                # Create DataFrame for visualization
                linear_importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': linear_importance
                }).sort_values('Importance', ascending=False)
                
                print("Top 20 features (Linear model):")
                print(linear_importance_df.head(20))
                
                # Store in all_importance
                all_importance['Linear'] = {feature: importance for feature, importance in zip(X.columns, linear_importance)}
                
                # Visualize linear model importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=linear_importance_df.head(20))
                plt.title('Feature Importance (Linear Model)')
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error fitting linear model: {e}")
                print("Skipping linear model importance analysis.")
            
            # Visualize tree-based model importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=tree_importance_df.head(20))
            plt.title('Feature Importance (Tree-based Model)')
            plt.tight_layout()
            plt.show()
            
            # 3. Compare importance rankings between models (if both available)
            if len(all_importance) > 1:
                print("\n3. Comparison of Feature Importance Rankings")
                
                # Get ranks for each model
                ranks = {}
                for model_name, importance in all_importance.items():
                    # Sort features by importance and get rankings
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    ranks[model_name] = {feature: idx+1 for idx, (feature, _) in enumerate(sorted_features)}
                
                # Create comparison DataFrame
                comparison_data = []
                for feature in X.columns:
                    feature_data = {'Feature': feature}
                    for model_name in all_importance.keys():
                        feature_data[f'{model_name} Rank'] = ranks[model_name].get(feature, np.nan)
                        feature_data[f'{model_name} Importance'] = all_importance[model_name].get(feature, np.nan)
                    comparison_data.append(feature_data)
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Calculate rank difference for the top features
                if 'Tree-based Rank' in comparison_df.columns and 'Linear Rank' in comparison_df.columns:
                    comparison_df['Rank Difference'] = np.abs(comparison_df['Tree-based Rank'] - comparison_df['Linear Rank'])
                    
                    # Sort by average rank
                    comparison_df['Average Rank'] = (comparison_df['Tree-based Rank'] + comparison_df['Linear Rank']) / 2
                    comparison_df = comparison_df.sort_values('Average Rank')
                    
                    print("Feature importance comparison (Top 20 by average rank):")
                    print(comparison_df.head(20))
                    
                    # Print features with largest disagreement
                    print("\nFeatures with largest disagreement in importance ranking:")
                    disagreement_df = comparison_df.sort_values('Rank Difference', ascending=False).head(10)
                    print(disagreement_df)
                    
                    # Visualize rank comparison for top features
                    try:
                        plt.figure(figsize=(14, 10))
                        
                        # Select top 20 features by average rank
                        top_features = comparison_df.head(20)['Feature'].tolist()
                        
                        # Create DataFrame in format for plotting
                        plot_data = []
                        for feature in top_features:
                            row = comparison_df[comparison_df['Feature'] == feature].iloc[0]
                            plot_data.append({
                                'Feature': feature,
                                'Model': 'Tree-based',
                                'Rank': row['Tree-based Rank']
                            })
                            plot_data.append({
                                'Feature': feature,
                                'Model': 'Linear',
                                'Rank': row['Linear Rank']
                            })
                        
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Plot
                        plt.figure(figsize=(10, 12))
                        ax = sns.barplot(x='Rank', y='Feature', hue='Model', data=plot_df)
                        plt.title('Feature Importance Rank Comparison (Lower is Better)')
                        plt.xlabel('Rank')
                        plt.ylabel('Feature')
                        plt.legend(title='Model Type')
                        
                        # Invert x-axis so that most important features (rank 1) are on the right
                        ax.invert_xaxis()
                        
                        plt.tight_layout()
                        plt.show()
                        
                    except Exception as e:
                        print(f"Error plotting rank comparison: {e}")
                
            else:
                print("\nImportance comparison skipped as only one model type is available.")
                
        except Exception as e:
            print(f"Error in feature importance calculation: {e}")
            import traceback
            traceback.print_exc()
    
    def run_full_analysis(self, importance=True):
        """Run all analysis methods in sequence."""
        print("=" * 80)
        print("DATASET ANALYSIS REPORT")
        print("=" * 80)
        
        print("\n1. BASIC INFORMATION")
        print("-" * 80)
        self.basic_info()
        
        print("\n2. MISSING VALUES ANALYSIS")
        print("-" * 80)
        self.missing_values_analysis()
        
        print("\n3. OUTLIER ANALYSIS")
        print("-" * 80)
        self.outlier_analysis()
        
        print("\n4. DISTRIBUTION ANALYSIS")
        print("-" * 80)
        self.distribution_analysis()
        
        print("\n5. CORRELATION ANALYSIS")
        print("-" * 80)
        self.correlation_analysis()
        
        print("\n6. CATEGORICAL FEATURES ANALYSIS")
        print("-" * 80)
        self.categorical_analysis()
        
        if self.target_col and importance:
            print("\n7. FEATURE IMPORTANCE")
            print("-" * 80)
            self.feature_importance()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)