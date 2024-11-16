# fraud_detection/api/endpoints.py

import joblib
import pandas as pd
import logging
from flask import jsonify, request
import os

# Initialize logging for the endpoints
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the credit fraud detection model
def load_credit_fraud_model():
    # Corrected model path based on folder structure
    credit_fraud_model_path = os.path.join('models', 'random_forest_model.pkl')
    try:
        credit_fraud_model = joblib.load(credit_fraud_model_path)
        logger.info("Credit Fraud Model loaded successfully.")
        return credit_fraud_model
    except Exception as e:
        logger.error(f"Error loading Credit Fraud Model: {e}")
        return None

# Load the general fraud detection model
def load_general_fraud_model():
    # Corrected model path based on folder structure
    general_fraud_model_path = os.path.join('models', 'random_forest_model.pkl')
    try:
        general_fraud_model = joblib.load(general_fraud_model_path)
        logger.info("General Fraud Model loaded successfully.")
        return general_fraud_model
    except Exception as e:
        logger.error(f"Error loading General Fraud Model: {e}")
        return None

# Define the specific feature sets for each model
general_fraud_model_features = [
    'user_id', 'purchase_value', 'device_id', 'source', 
    'browser', 'sex', 'age', 'country', 
    'transaction_frequency', 'transaction_velocity', 
    'hour_of_day', 'day_of_week'
]

credit_fraud_model_features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']

# Helper function to ensure input data has required columns
def validate_input(data, model):
    if model is None:
        return False, "Model is not loaded."
    
    required_columns = credit_fraud_model_features if model == credit_fraud_model else general_fraud_model_features
    if set(required_columns).issubset(data.keys()):
        return True, None
    else:
        missing_cols = set(required_columns) - set(data.keys())
        return False, f"Missing columns: {missing_cols}"

# Define a route for predictions using the credit fraud model
def predict_credit_fraud():
    # Load model
    credit_fraud_model = load_credit_fraud_model()
    if credit_fraud_model is None:
        return jsonify({'error': "Error loading Credit Fraud Model."}), 500

    # Get data from the request
    data = request.get_json()

    # Validate input data
    is_valid, error_msg = validate_input(data, credit_fraud_model)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Ensure the data is in the correct format
    input_df = pd.DataFrame([data])

    # Make predictions using the credit fraud model
    try:
        predictions = credit_fraud_model.predict(input_df)
        logger.info("Prediction made using Credit Fraud Model.")
    except Exception as e:
        logger.error(f"Error making prediction with Credit Fraud Model: {e}")
        return jsonify({'error': str(e)}), 500

    # Return predictions as a JSON response
    return jsonify(predictions.tolist())

# Define a route for predictions using the general fraud model
def predict_general_fraud():
    # Load model
    general_fraud_model = load_general_fraud_model()
    if general_fraud_model is None:
        return jsonify({'error': "Error loading General Fraud Model."}), 500

    # Get data from the request
    data = request.get_json()

    # Validate input data
    is_valid, error_msg = validate_input(data, general_fraud_model)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Ensure the data is in the correct format
    input_df = pd.DataFrame([data])

    # Make predictions using the general fraud model
    try:
        predictions = general_fraud_model.predict(input_df)
        logger.info("Prediction made using General Fraud Model.")
    except Exception as e:
        logger.error(f"Error making prediction with General Fraud Model: {e}")
        return jsonify({'error': str(e)}), 500

    # Return predictions as a JSON response
    return jsonify(predictions.tolist())
