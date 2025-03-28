# General settings
random_state = 42

# Classification model parameters
classification_params = {
    'logistic': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1],
        'class_weight': [None, 'balanced']
    },
    'random_forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 2, 3]  # For handling class imbalance
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'num_leaves': [31, 50, 100],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 2, 3]  # For handling class imbalance
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.3],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128],
        'scale_pos_weight': [1, 2, 3]  # For handling class imbalance
    },
    'decision_tree': {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Class imbalance handling
class_imbalance_settings = {
    'method': 'SMOTE',  # Options: 'SMOTE', 'ADASYN', 'RandomUnderSampler', etc.
    'sampling_strategy': 'auto',  # Can be a float or a dict
    'k_neighbors': 5,  # For SMOTE and ADASYN
    'random_state': random_state
}

# Model selection
available_models = [
    'logistic',
    'svm',
    'random_forest',
    'xgboost',
    'lightgbm',
    'catboost',
    'decision_tree',
    'knn',
    'naive_bayes',
    'gradient_boosting'
]

# Default settings for classification
default_classification_settings = {
    'cv': 5,
    'n_repeats' : 5,
    'test_size': 0.2,
    'n_splits': 3,
    'stratify': None,
    'scoring': 'roc_auc',
    'n_iter': 20,
    'optimization_method': 'random',
    'n_jobs': -1,
    'models_to_train': available_models,  # Default to all available models
    'class_imbalance': class_imbalance_settings  # Include class imbalance settings
}