# fraud_detection/model_training.py
import time
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.optimizers import Adam
from utils.logging_util import setup_logger
import mlflow
import mlflow.sklearn


# Setup logger
logger = setup_logger('../logs/training.log')



def prepare_large_data(data_path, target_column, test_size=0.2, scale_data=True, scaler_path=None):
    """
    Data preparation function for datasets with categorical and numerical features.

    Parameters:
    - data_path (str): Path to the dataset file (CSV).
    - target_column (str): Name of the target column.
    - test_size (float): Fraction of data to reserve for testing.
    - scale_data (bool): Whether to standardize features.
    - scaler_path (str): Path to save the scaler if scaling is applied.

    Returns:
    - dict: Contains 'X_train', 'X_test', 'y_train', 'y_test', and 'scaler' (if applicable).
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")

        # Handle time-related features
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            df['time_to_purchase'] = (pd.to_datetime(df['purchase_time']) - 
                                       pd.to_datetime(df['signup_time'])).dt.total_seconds()
            df.drop(columns=['signup_time', 'purchase_time'], inplace=True)
            logger.info("Calculated 'time_to_purchase' and dropped 'signup_time' and 'purchase_time'.")

        # Encode high-cardinality 'country' column using frequency encoding
        if 'country' in df.columns:
            df['country_encoded'] = df['country'].map(df['country'].value_counts())
            df.drop(columns=['country'], inplace=True)
            logger.info("Encoded 'country' using frequency encoding.")

        # Encode other categorical columns
        categorical_columns = ['source', 'browser', 'sex']
        for col in categorical_columns:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
                logger.info(f"Encoded column: {col} using one-hot encoding.")

        # Drop irrelevant columns (removing device_id and ip_address)
        drop_cols = ['user_id', 'device_id', 'ip_address']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        logger.info(f"Dropped irrelevant columns: {drop_cols}")

        # Define features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info(f"Features (X) and target (y) defined. Target column: {target_column}")

        # Capture feature names before scaling
        feature_names = X.columns.tolist()  # Store feature names before scaling

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info(f"Data split into train and test sets with test size = {test_size}.")

        # Scale the data if needed
        scaler = None
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Standardized the data.")

            # Save the scaler if scaler_path is specified
            if scaler_path:
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")

        # Return feature names and scaled data
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'scaler': scaler, 'feature_names': feature_names}

    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise


def prepare_data(data_path, target_column, test_size=0.2, scale_data=True, scaler_path=None):
    """
    General data preparation function for datasets with categorical and numerical features.

    Parameters:
    - data_path (str): Path to the dataset file (CSV).
    - target_column (str): Name of the target column.
    - test_size (float): Fraction of data to reserve for testing.
    - scale_data (bool): Whether to standardize features.
    - scaler_path (str): Path to save the scaler if scaling is applied.

    Returns:
    - dict: Contains 'X_train', 'X_test', 'y_train', 'y_test', and 'scaler' (if applicable).
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")

        # Drop unnecessary columns (if specific to the dataset)
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            df.drop(columns=['signup_time', 'purchase_time'], inplace=True)

        # Handle categorical columns by encoding them
        categorical_columns = ['device_id', 'source', 'browser', 'sex']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                logger.info(f"Encoded column: {col}")

        # Define features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        feature_names = X.columns.tolist()

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info("Data successfully split into train and test sets.")

        # Scale the data if needed
        scaler = None
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Standardized the data.")

            # Save the scaler if scaler_path is specified
            if scaler_path:
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")

        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'scaler': scaler,'feature_names': feature_names }

    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise
    
# 
def prepare_credit_card_data(data_path, target_column, test_size=0.2, scale_data=True, scaler_path=None):
    """
    Prepare credit card fraud data for training and testing.

    Parameters:
    - data_path (str): Path to the dataset file (CSV).
    - target_column (str): Name of the target column indicating fraud.
    - test_size (float): Fraction of data to reserve for testing.
    - scale_data (bool): Whether to standardize features.
    - scaler_path (str): Path to save the scaler if scaling is applied.

    Returns:
    - dict: Contains 'X_train', 'X_test', 'y_train', 'y_test', and 'scaler' (if applicable).
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")

        # Check for missing values and handle them
        if df.isnull().sum().sum() > 0:
            logger.warning("Dataset contains missing values. Filling with mean values.")
            df.fillna(df.mean(), inplace=True)

        # Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        feature_names = X.columns.tolist()

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info("Data successfully split into training and testing sets.")

        # Scale the data if needed
        scaler = None
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Feature scaling applied.")

            # Save the scaler if scaler_path is specified
            if scaler_path:
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")

        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'scaler': scaler ,'feature_names': feature_names}

    except Exception as e:
        logger.error(f"Error in prepare_credit_card_data: {e}")
        raise

def train_and_log_model(model, X_train, X_test, y_train, y_test, model_name, experiment_name="FraudDetection"):
    """
    Train a model, log it with MLflow, and save the best-performing model.

    Parameters:
        - model: Initialized ML model.
        - X_train, X_test, y_train, y_test: Train/test splits.
        - model_name: Name of the model (for MLflow).
        - experiment_name: MLflow experiment name.

    Returns:
        - dict: Contains metrics and trained model.
    """
    try:
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        # Start a new MLflow run
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")

            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                roc_auc = None

            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info(f"{model_name} training complete.")

            # Log parameters, metrics, and model to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("training_time", training_time)
            mlflow.log_metrics({
                "roc_auc": roc_auc,
                "precision": report['1']['precision'],
                "recall": report['1']['recall'],
                "f1_score": report['1']['f1-score']
            })

            # Log the model to MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Log additional artifacts (e.g., scaler if applicable)
            logger.info(f"{model_name} logged to MLflow.")

            return {"model": model, "roc_auc": roc_auc, "report": report, "training_time": training_time}

    except Exception as e:
        logger.error(f"Error training and logging model: {e}")
        raise

def train_and_select_best_model(models, X_train, X_test, y_train, y_test, feature_names):
    """
    Train multiple models and select the best one based on ROC-AUC.

    Parameters:
        - models: Dictionary of model names and initialized model objects.
        - X_train, X_test, y_train, y_test: Train/test splits.
        - feature_names: List of feature names (for explainability).

    Returns:
        - dict: Best model details (name, object, metrics, and feature names).
    """
    best_model = None
    best_roc_auc = 0
    best_model_details = {}

    for model_name, model in models.items():
        logger.info(f"Training model: {model_name}")
        result = train_and_log_model(model, X_train, X_test, y_train, y_test, model_name)

        # Compare models by ROC-AUC
        if result["roc_auc"] > best_roc_auc:
            best_roc_auc = result["roc_auc"]
            best_model = model
            best_model_details = {
                "name": model_name,
                "roc_auc": best_roc_auc,
                "model": best_model,
                "feature_names": feature_names  # Include feature names
            }

    logger.info(f"Best model: {best_model_details['name']} with ROC-AUC: {best_roc_auc}")
    return best_model_details
# def train_and_save_model(model, X_train, X_test, y_train, y_test, model_path, scaler=None, scaler_path=None):
#     """Train and save a model and its scaler to disk."""
#     try:
#         logger.info(f"Training model: {model.__class__.__name__}.")
#         start_time = time.time()
#         model.fit(X_train, y_train)
#         training_time = time.time() - start_time

#         # Save the model to disk
#         joblib.dump(model, model_path)
#         logger.info(f"Model saved to {model_path}.")

#         # If the scaler is provided, save it
#         if scaler is not None and scaler_path:
#             joblib.dump(scaler, scaler_path)
#             logger.info(f"Scaler saved to {scaler_path}.")

#         # Evaluate the model
#         y_pred = model.predict(X_test)
#         if hasattr(model, "predict_proba"):
#             y_pred_proba = model.predict_proba(X_test)[:, 1]
#             roc_auc = roc_auc_score(y_test, y_pred_proba)
#         else:
#             roc_auc = None

#         report = classification_report(y_test, y_pred)
#         logger.info("Model training and evaluation completed successfully.")
#         return {"model": model, "training_time": training_time, "roc_auc": roc_auc, "report": report}
#     except Exception as e:
#         logger.error(f"Error training model: {e}")
#         raise


# # Individual training functions for various models
# def train_logistic_regression(X_train, X_test, y_train, y_test, model_path, scaler=None, scaler_path=None):
#     model = LogisticRegression(max_iter=2000, class_weight="balanced")
#     return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path, scaler, scaler_path)


# def train_decision_tree(X_train, X_test, y_train, y_test, model_path, scaler=None, scaler_path=None):
#     model = DecisionTreeClassifier()
#     return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path, scaler, scaler_path)


# def train_random_forest(X_train, X_test, y_train, y_test, model_path, scaler=None, scaler_path=None):
#     model = RandomForestClassifier(class_weight="balanced")
#     return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path, scaler, scaler_path)


# def train_gradient_boosting(X_train, X_test, y_train, y_test, model_path, scaler=None, scaler_path=None):
#     model = GradientBoostingClassifier(n_estimators=100)
#     return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path, scaler, scaler_path)

