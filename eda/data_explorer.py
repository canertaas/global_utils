import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import StandardScaler
import warnings
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
                sample = sample.sample(5000, random_state=42)
            
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
        corr_matrix = self.df[self.numerical_cols].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
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
                print(f"Top 10 categories:")
                print(value_counts.head(10))
                print(f"(Showing 10/{unique_count} categories)")
    
    def feature_importance(self):
        """Calculate feature importance if target column is provided."""
        if not self.target_col:
            print("No target column provided for feature importance analysis.")
            return
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Prepare data
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            
            # Handle categorical features
            X = pd.get_dummies(X, drop_first=True)
            
            # Choose model based on target type
            if self.df[self.target_col].nunique() < 20:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Fit model
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("Feature Importance:")
            print(feature_importance.head(20))
            
            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in feature importance calculation: {e}")
    
    def run_full_analysis(self):
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
        
        if self.target_col:
            print("\n7. FEATURE IMPORTANCE")
            print("-" * 80)
            self.feature_importance()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)