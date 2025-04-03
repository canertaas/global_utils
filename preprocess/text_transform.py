from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from .pca import PCAApplication

class TextProcessor:
    """
    A general utility class for processing and extracting features from text data.
    Handles both training and test datasets with methods for text embedding,
    TF-IDF processing, and feature engineering.
    """
    
    def __init__(self):
        """Initialize the TextProcessor class."""
        self.tfidf = None
        self.pca = None
        self.embedding_model = None
        self.fitted = False
    
    def setup_embedding_model(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the sentence embedding model.
        
        Parameters:
        -----------
        model_name : str, default='all-MiniLM-L6-v2'
            Name of the pre-trained sentence transformer model to use
            
        Returns:
        --------
        self : TextFeatureProcessor
            Returns self for method chaining
        """
        self.embedding_model = SentenceTransformer(model_name)
        return self
    
    def create_embeddings(self, data, column_name, is_train=True):
        """
        Create embeddings for text data using sentence transformer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the column to embed
        column_name : str
            Name of the column to create embeddings for
        is_train : bool, default=True
            Whether this is training data (for logging purposes)
            
        Returns:
        --------
        numpy.ndarray
            Array of embeddings
        """
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        text_list = data[column_name].tolist()
        dataset_type = "train" if is_train else "test"
        print(f"Creating embeddings for {column_name} ({dataset_type})...")
        
        embeddings = self.embedding_model.encode(text_list)
        print(f"Embeddings created with shape: {embeddings.shape}")
        
        return embeddings
    
    def create_tfidf(self, data, column_name, max_features=100, is_train=True):
        """
        Create TF-IDF features for text data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the column to process
        column_name : str
            Name of the column to create TF-IDF features for
        max_features : int, default=100
            Maximum number of features to extract
        is_train : bool, default=True
            Whether this is training data (affects fitting behavior)
            
        Returns:
        --------
        numpy.ndarray
            Array of TF-IDF features
        """
        if is_train or self.tfidf is None:
            self.tfidf = TfidfVectorizer(max_features=max_features)
            X_tfidf = self.tfidf.fit_transform(data[column_name]).toarray()
            self.fitted = True
        else:
            X_tfidf = self.tfidf.transform(data[column_name]).toarray()
        
        dataset_type = "train" if is_train else "test"
        print(f"TF-IDF features created for {column_name} ({dataset_type}) with shape: {X_tfidf.shape}")
        
        return X_tfidf
    
    def apply_pca(self, train_data, test_data=None, n_components=5, variance_threshold=0.95):
        """
        Apply PCA dimensionality reduction to data.
        Fits on train data and transforms both train and test data.
        
        Parameters:
        -----------
        train_data : numpy.ndarray
            Training data to fit PCA on and transform
        test_data : numpy.ndarray, default=None
            Test data to transform using the fitted PCA
        n_components : int, default=5
            Number of principal components to keep
        variance_threshold : float, default=0.95
            Variance threshold for optimal components
            
        Returns:
        --------
        dict
            Dictionary containing transformed train and test data
        """
        # Initialize PCA application
        self.pca = PCAApplication(n_components=n_components, variance_threshold=variance_threshold)
        
        # Fit on train data and transform
        train_transformed = self.pca.fit_transform(train_data)
        print(f"PCA fitted on train data with shape: {train_data.shape}")
        print(f"Train data transformed with output shape: {train_transformed.shape}")
        
        result = {'train': train_transformed}
        
        # Transform test data if provided
        if test_data is not None:
            test_transformed = self.pca.transform(test_data)
            print(f"Test data transformed with output shape: {test_transformed.shape}")
            result['test'] = test_transformed
        
        return result
    
    def extract_text_features(self, data, column_name, is_train=True):
        """
        Extract simple text features like length and word count.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the text column
        column_name : str
            Name of the column to extract features from
        is_train : bool, default=True
            Whether this is training data (for logging purposes)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new text features added
        """
        data_copy = data.copy()
        
        # Add text length feature
        data_copy[f'{column_name}_len'] = data_copy[column_name].apply(lambda x: len(str(x)))
        
        # Add word count feature
        data_copy[f'{column_name}_word_count'] = data_copy[column_name].apply(
            lambda x: len(str(x).split()))
        
        # Add has_number feature
        data_copy[f'{column_name}_has_number'] = data_copy[column_name].str.contains(r'\d').astype(int)
        
        # Add has_question feature
        data_copy[f'{column_name}_has_question'] = data_copy[column_name].str.contains(r'\?').astype(int)
        
        # Add has_exclamation feature
        data_copy[f'{column_name}_has_exclamation'] = data_copy[column_name].str.contains(r'\!').astype(int)
        
        # Add uppercase ratio feature
        data_copy[f'{column_name}_uppercase_ratio'] = data_copy[column_name].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
        
        dataset_type = "train" if is_train else "test"
        print(f"Text features extracted for {column_name} ({dataset_type})")
        
        return data_copy
    
    def process_features(self, train_data, test_data, 
                         general_feature_columns, embedding_columns, 
                         tfidf_columns, n_components=5, variance_threshold=0.95):
        """
        Process both train and test datasets together.
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Training dataframe
        test_data : pandas.DataFrame
            Test dataframe
        general_feature_columns : list
            List of general feature column names to process
        embedding_columns : list
            List of embedding column names to process
        tfidf_columns : list
            List of TF-IDF column names to process
        n_components : int, default=5
            Number of principal components to keep
            
        Returns:
        --------
        dict
            Dictionary containing processed train and test data with features
        """
        # Process train data text features
        print("Extracting text features for training data...")

        train_processed = train_data.copy()
        test_processed = test_data.copy()

        if general_feature_columns:
            for col in general_feature_columns:
                train_processed = self.extract_text_features(train_processed, col, is_train=True)
            
            # Process test data text features
            print("\nExtracting text features for test data...")
            for col in general_feature_columns:
                test_processed = self.extract_text_features(test_processed, col, is_train=False)
        
        feature_dict = {}
        
        # Process embeddings
        if embedding_columns:
            for col in embedding_columns:
                print(f"\nProcessing embeddings for {col}...")
                train_embeddings = self.create_embeddings(train_processed, col, is_train=True)
            test_embeddings = self.create_embeddings(test_processed, col, is_train=False)
            feature_dict[f'{col}_embeddings'] = {
                'train': train_embeddings,
                'test': test_embeddings
            }
        
        # Process TF-IDF features
        if tfidf_columns:
            for col in tfidf_columns:
                print(f"\nProcessing TF-IDF for {col}...")
                train_tfidf = self.create_tfidf(train_processed, col, is_train=True)
            test_tfidf = self.create_tfidf(test_processed, col, is_train=False)
            
            feature_dict[f'{col}_tfidf'] = {
                'train': train_tfidf,
                'test': test_tfidf
            }
            
            # Apply PCA to TF-IDF features
            print(f"\nApplying PCA to {col} TF-IDF features...")
            tfidf_pca_results = self.apply_pca(train_tfidf, test_tfidf, n_components=n_components, 
                                               variance_threshold=variance_threshold)
            feature_dict[f'{col}_tfidf_pca'] = tfidf_pca_results

        # Return combined results
        return {
            'train_data': train_processed,
            'test_data': test_processed,
            'features': feature_dict
        } 