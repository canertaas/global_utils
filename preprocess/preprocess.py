import pandas as pd 
from ..config import random_state

class DataPreprocessor:
    """
    A class to apply preprocessing steps to a dataset, including handling missing values,
    creating is_na indicator columns, and other common preprocessing tasks.
    """
    
    def __init__(self, df=None):
        """
        Initialize the DataPreprocessor with an optional dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            The dataframe to preprocess
        """
        self.df = df.copy() if df is not None else None
        self.na_columns = []
        self.preprocessing_steps = []
    
    def set_dataframe(self, df):
        """Set the dataframe to preprocess."""
        self.df = df.copy()
        return self
    
    def create_na_indicators(self, columns=None):
        """
        Create binary indicator columns for missing values.
        
        Parameters:
        -----------
        columns : list or None
            List of columns to create indicators for. If None, creates for all columns with NAs.
        
        Returns:
        --------
        self : DataPreprocessor
            Returns self for method chaining
        """
        if self.df is None:
            raise ValueError("DataFrame not set. Use set_dataframe() first.")
        
        # If columns not specified, find all columns with missing values
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].isna().any()]
        
        # Create indicator columns
        for col in columns:
            if col in self.df.columns:
                na_col_name = f"is_na_{col}"
                self.df[na_col_name] = self.df[col].isna().astype(int)
                self.na_columns.append(na_col_name)
                
        self.preprocessing_steps.append("Created NA indicators")
        return self
    
    def fill_missing_values(self, strategy='mean', columns=None, fill_value=None):
        """
        Fill missing values in the dataframe.
        
        Parameters:
        -----------
        strategy : str, default='mean'
            Strategy to use for filling missing values: 'mean', 'median', 'mode', 'constant'
        columns : list or None
            List of columns to fill. If None, fills all columns with NAs.
        fill_value : any, default=None
            Value to use when strategy is 'constant'
            
        Returns:
        --------
        self : DataPreprocessor
            Returns self for method chaining
        """
        if self.df is None:
            raise ValueError("DataFrame not set. Use set_dataframe() first.")
        
        # If columns not specified, find all columns with missing values
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].isna().any()]
        
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == 'constant':
                    self.df[col] = self.df[col].fillna(fill_value)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
        
        self.preprocessing_steps.append(f"Filled missing values using {strategy}")
        return self
    
    def drop_columns(self, columns):
        """Drop specified columns from the dataframe."""
        if self.df is None:
            raise ValueError("DataFrame not set. Use set_dataframe() first.")
        
        self.df = self.df.drop(columns=columns, errors='ignore')
        self.preprocessing_steps.append(f"Dropped columns: {columns}")
        return self
    
    def encode_categorical(self, columns=None, method='one-hot', random_state=random_state):
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        columns : list or None
            List of columns to encode. If None, encodes all object/category columns.
        method : str, default='one-hot'
            Encoding method: 'one-hot' or 'label'
            
        Returns:
        --------
        self : DataPreprocessor
            Returns self for method chaining
        """
        if self.df is None:
            raise ValueError("DataFrame not set. Use set_dataframe() first.")
        
        # If columns not specified, find all categorical columns
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'one-hot':
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=False)
            self.preprocessing_steps.append(f"One-hot encoded columns: {columns}")
        elif method == 'label':
            for col in columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('category').cat.codes
            self.preprocessing_steps.append(f"Label encoded columns: {columns}")
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return self
    
    def get_processed_data(self):
        """Return the processed dataframe."""
        if self.df is None:
            raise ValueError("DataFrame not set. Use set_dataframe() first.")
        
        return self.df.copy()
    
    def get_preprocessing_summary(self):
        """Return a summary of preprocessing steps applied."""
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(self.preprocessing_steps)])