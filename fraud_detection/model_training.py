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
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder 

from utils.logging_util import setup_logger

# Setup logger

logger = setup_logger('../logs/training.log')


def prepare_data(data_path, target_column, test_size=0.2, scale_data=True):
    """ Prepare data for training and testing, handling timestamps and categorical variables. """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        logging.info(f"Loaded data from {data_path}")

        # Convert timestamp columns to datetime
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Calculate time difference between signup and purchase (in seconds)
        df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

        # Drop the original timestamp columns
        df.drop(columns=['signup_time', 'purchase_time'], inplace=True)

        # Handle categorical columns by encoding them
        categorical_columns = ['device_id', 'source', 'browser', 'sex']
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            logging.info(f"Encoded column: {col}")

        # Define features (X) and target (y)
        X = df.drop(columns=[target_column])  # Features (everything except the target column)
        y = df[target_column]  # Target

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logging.info("Data successfully split into train and test sets.")

        # Scale the data if needed
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logging.info("Standardized the data.")

        # Return the prepared data
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise

def train_and_save_model(model, X_train, X_test, y_train, y_test, model_path):
    """
    Train and save a model to disk.

    Parameters:
        model: The machine learning model to train.
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        model_path (str): Path to save the trained model.

    Returns:
        dict: A dictionary containing model performance metrics and training time.
    """
    try:
        logging.info(f"Training model: {model.__class__.__name__}.")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Save the model to disk
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None

        report = classification_report(y_test, y_pred)
        logging.info("Model training and evaluation completed successfully.")
        return {"model": model, "training_time": training_time, "roc_auc": roc_auc, "report": report}
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise


# Individual training functions
def train_logistic_regression(X_train, X_test, y_train, y_test, model_path):
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_decision_tree(X_train, X_test, y_train, y_test, model_path):
    model = DecisionTreeClassifier()
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_random_forest(X_train, X_test, y_train, y_test, model_path):
    model = RandomForestClassifier(class_weight="balanced")
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_gradient_boosting(X_train, X_test, y_train, y_test, model_path):
    model = GradientBoostingClassifier(n_estimators=100)
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


# Example usage in model_train.ipynb:
# ------------------------------------
# data = prepare_data("data/datas.csv", target_column="fraud_class")
# results = train_logistic_regression(
#     data["X_train_scaled"], data["X_test_scaled"], data["y_train"], data["y_test"], "models/logistic_regression_model.pkl"
# )
# print(results)

