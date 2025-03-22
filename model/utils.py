import os
import joblib
import h2o
import warnings
warnings.filterwarnings('ignore')

class ModelUtils:
    """
    Utility class for model persistence operations like saving and loading models.
    """
    
    @staticmethod
    def save_model(model, path, model_name):
        """
        Save a model to disk.
        
        Parameters:
        -----------
        model : object
            The model to save
        path : str
            Directory path to save the model
        model_name : str
            Name of the model file
            
        Returns:
        --------
        str
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Determine full path
        full_path = os.path.join(path, model_name)
        
        # Save the model
        joblib.dump(model, full_path)
        
        return full_path
    
    @staticmethod
    def load_model(path):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        path : str
            Path to the model file
            
        Returns:
        --------
        object
            The loaded model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        return joblib.load(path)
    
    @staticmethod
    def save_h2o_model(model, path, model_name):
        """
        Save an H2O model to disk.
        
        Parameters:
        -----------
        model : h2o.model
            The H2O model to save
        path : str
            Directory path to save the model
        model_name : str
            Name of the model directory
            
        Returns:
        --------
        str
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Determine full path
        full_path = os.path.join(path, model_name)
        
        # Save the model
        h2o.save_model(model, path=full_path)
        
        return full_path
    
    @staticmethod
    def load_h2o_model(path):
        """
        Load an H2O model from disk.
        
        Parameters:
        -----------
        path : str
            Path to the H2O model directory
            
        Returns:
        --------
        h2o.model
            The loaded H2O model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"H2O model directory not found at {path}")
        
        return h2o.load_model(path)