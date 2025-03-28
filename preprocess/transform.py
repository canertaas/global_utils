import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import QuantileTransformer, Normalizer
import warnings
from ..config import random_state
warnings.filterwarnings('ignore')

class DataTransformer:
    """
    A comprehensive class for applying various transformations to data.
    Supports both column-wise and dataset-wide transformations.
    
    Features:
    - Mathematical transformations (log, sqrt, etc.)
    - Scaling methods (standard, minmax, etc.)
    - Normalization methods
    - Power transformations
    - Custom transformations
    """
    
    def __init__(self):
        """Initialize transformer with empty state"""
        self.fitted_scalers = {}
        self.fitted_transformers = {}
        self.transformation_history = []
    
    def _validate_input(self, data):
        """Validate input data type and convert if necessary"""
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        else:
            raise ValueError("Input must be pandas Series, DataFrame, list, or numpy array")
    
    def log_transform(self, data, columns=None, base=np.e, handle_zeros=True, epsilon=1e-8):
        """
        Apply logarithmic transformation.
        
        Parameters:
        -----------
        data : pandas.Series, pandas.DataFrame, or array-like
            Data to transform
        columns : list, default=None
            Columns to transform if data is DataFrame
        base : float, default=np.e
            Base of logarithm
        handle_zeros : bool, default=True
            Whether to handle zero values by adding epsilon
        epsilon : float, default=1e-8
            Small constant to add when handling zeros
            
        Returns:
        --------
        transformed : pandas.Series or pandas.DataFrame
            Transformed data
        """
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                values = result[col]
                if handle_zeros:
                    values = values + epsilon
                result[col+'log'] = np.log(values) / np.log(base)
            
            self.transformation_history.append(('log', columns, {'base': base}))
            return result
        
        else:  # Series or array
            if handle_zeros:
                data = data + epsilon
            transformed = np.log(data) / np.log(base)
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def sqrt_transform(self, data, columns=None):
        """Apply square root transformation"""
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                result[col+'sqrt'] = np.sqrt(np.abs(result[col])) * np.sign(result[col])
            
            self.transformation_history.append(('sqrt', columns, {}))
            return result
        
        else:  # Series or array
            transformed = np.sqrt(np.abs(data)) * np.sign(data)
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def box_cox_transform(self, data, columns=None):
        """Apply Box-Cox transformation"""
        data = self._validate_input(data)
        transformer = PowerTransformer(method='box-cox', standardize=False)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                # Box-Cox requires positive values
                min_val = result[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    result[col] = result[col] + shift
                
                result[col+'box_cox'] = transformer.fit_transform(result[col].values.reshape(-1, 1)).ravel()
            
            self.transformation_history.append(('box-cox', columns, {}))
            return result
        
        else:  # Series or array
            # Handle negative values
            min_val = np.min(data)
            if min_val <= 0:
                shift = abs(min_val) + 1
                data = data + shift
            
            transformed = transformer.fit_transform(np.array(data).reshape(-1, 1)).ravel()
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def yeo_johnson_transform(self, data, columns=None):
        """Apply Yeo-Johnson transformation"""
        data = self._validate_input(data)
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                result[col+'yeo_johnson'] = transformer.fit_transform(result[col].values.reshape(-1, 1)).ravel()
            
            self.transformation_history.append(('yeo-johnson', columns, {}))
            return result
        
        else:  # Series or array
            transformed = transformer.fit_transform(np.array(data).reshape(-1, 1)).ravel()
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def standard_scale(self, data, columns=None):
        """Apply StandardScaler transformation"""
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                scaler = StandardScaler()
                result[col] = scaler.fit_transform(result[col].values.reshape(-1, 1)).ravel()
                self.fitted_scalers[f'standard_{col}'] = scaler
            
            self.transformation_history.append(('standard_scale', columns, {}))
            return result
        
        else:  # Series or array
            scaler = StandardScaler()
            transformed = scaler.fit_transform(np.array(data).reshape(-1, 1)).ravel()
            self.fitted_scalers['standard'] = scaler
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def minmax_scale(self, data, columns=None, feature_range=(0, 1)):
        """Apply MinMaxScaler transformation"""
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                scaler = MinMaxScaler(feature_range=feature_range)
                result[col] = scaler.fit_transform(result[col].values.reshape(-1, 1)).ravel()
                self.fitted_scalers[f'minmax_{col}'] = scaler
            
            self.transformation_history.append(('minmax_scale', columns, {'feature_range': feature_range}))
            return result
        
        else:  # Series or array
            scaler = MinMaxScaler(feature_range=feature_range)
            transformed = scaler.fit_transform(np.array(data).reshape(-1, 1)).ravel()
            self.fitted_scalers['minmax'] = scaler
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def robust_scale(self, data, columns=None):
        """Apply RobustScaler transformation"""
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                scaler = RobustScaler()
                result[col] = scaler.fit_transform(result[col].values.reshape(-1, 1)).ravel()
                self.fitted_scalers[f'robust_{col}'] = scaler
            
            self.transformation_history.append(('robust_scale', columns, {}))
            return result
        
        else:  # Series or array
            scaler = RobustScaler()
            transformed = scaler.fit_transform(np.array(data).reshape(-1, 1)).ravel()
            self.fitted_scalers['robust'] = scaler
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def quantile_transform(self, data, columns=None, n_quantiles=1000, output_distribution='normal'):
        """Apply QuantileTransformer transformation"""
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                transformer = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution=output_distribution,
                    random_state=random_state
                )
                result[col] = transformer.fit_transform(result[col].values.reshape(-1, 1)).ravel()
                self.fitted_transformers[f'quantile_{col}'] = transformer
            
            self.transformation_history.append(
                ('quantile_transform', columns, 
                 {'n_quantiles': n_quantiles, 'output_distribution': output_distribution})
            )
            return result
        
        else:  # Series or array
            transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution=output_distribution,
                random_state=random_state
            )
            transformed = transformer.fit_transform(np.array(data).reshape(-1, 1)).ravel()
            self.fitted_transformers['quantile'] = transformer
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def normalize(self, data, columns=None, norm='l2'):
        """Apply normalization"""
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            normalizer = Normalizer(norm=norm)
            result[columns] = normalizer.fit_transform(result[columns])
            self.fitted_transformers['normalizer'] = normalizer
            
            self.transformation_history.append(('normalize', columns, {'norm': norm}))
            return result
        
        else:  # Series or array
            normalizer = Normalizer(norm=norm)
            transformed = normalizer.fit_transform(np.array(data).reshape(1, -1))[0]
            self.fitted_transformers['normalizer'] = normalizer
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def custom_transform(self, data, columns=None, func=None):
        """
        Apply custom transformation function.
        
        Parameters:
        -----------
        data : pandas.Series, pandas.DataFrame, or array-like
            Data to transform
        columns : list, default=None
            Columns to transform if data is DataFrame
        func : callable
            Custom transformation function
            
        Returns:
        --------
        transformed : pandas.Series or pandas.DataFrame
            Transformed data
        """
        if func is None:
            raise ValueError("Must provide a transformation function")
        
        data = self._validate_input(data)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            result = data.copy()
            for col in columns:
                result[col] = func(result[col])
            
            self.transformation_history.append(('custom', columns, {'func': func.__name__}))
            return result
        
        else:  # Series or array
            transformed = func(data)
            return pd.Series(transformed, index=data.index) if hasattr(data, 'index') else transformed
    
    def get_transformation_history(self):
        """Get the history of applied transformations"""
        return pd.DataFrame(self.transformation_history, 
                          columns=['transformation', 'columns', 'parameters'])