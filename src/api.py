import joblib  # Make sure you import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the credit fraud detection model
credit_fraud_model_path = 'model/Credit Card Fraud Detection with Random Forest_random_forest_model.pkl'
try:
    credit_fraud_model = joblib.load(credit_fraud_model_path)  # Use joblib to load the model
    logger.info("Credit Fraud Model loaded successfully.")
    logger.info(f"Credit Fraud Model type: {type(credit_fraud_model)}")  # Log the type of the model

    # Check if credit_fraud_model is a valid model
    if not hasattr(credit_fraud_model, 'predict'):
        logger.error("Credit Fraud Model is not a valid model object.")
except Exception as e:
    logger.error(f"Error loading Credit Fraud Model: {e}")
    credit_fraud_model = None  # Set to None if loading fails

# Load the general fraud detection model
general_fraud_model_path = 'model/Fraud Detection with Random Forest_random_forest_model.pkl'
try:
    general_fraud_model = joblib.load(general_fraud_model_path)  # Use joblib to load the model
    logger.info("General Fraud Model loaded successfully.")
    logger.info(f"General Fraud Model type: {type(general_fraud_model)}")  # Log the type of the model

    # Check if general_fraud_model is a valid model
    if not hasattr(general_fraud_model, 'predict'):
        logger.error("General Fraud Model is not a valid model object.")
except Exception as e:
    logger.error(f"Error loading General Fraud Model: {e}")
    general_fraud_model = None  # Set to None if loading fails

# Define the specific feature sets for each model
general_fraud_model_features = [
    'user_id', 'purchase_value', 'device_id', 'source', 
    'browser', 'sex', 'age', 'country', 
    'transaction_frequency', 'transaction_velocity', 
    'hour_of_day', 'day_of_week'
]

credit_fraud_model_features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']  # Make sure these match the training features

# Helper function to ensure input data has required columns
def validate_input(data, model):
    if model is None:
        return False, "Model is not loaded."
    
    # Use credit_fraud_model_features for validation if using credit fraud model
    required_columns = credit_fraud_model_features if model == credit_fraud_model else general_fraud_model_features
    if set(required_columns).issubset(data.keys()):
        return True, None
    else:
        missing_cols = set(required_columns) - set(data.keys())
        return False, f"Missing columns: {missing_cols}"

# Define a route for predictions using the credit fraud model
@app.route('/predict/credit_fraud', methods=['POST'])
def predict_credit_fraud():
    # Get data from the request
    data = request.get_json()
    
    # Validate input data
    is_valid, error_msg = validate_input(data, credit_fraud_model)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Ensure the data is in the correct format
    input_df = pd.DataFrame([data])  # Wrap data in a list to ensure it is treated as a single row

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
@app.route('/predict/general_fraud', methods=['POST'])
def predict_general_fraud():
    # Get data from the request
    data = request.get_json()

    # Validate input data
    is_valid, error_msg = validate_input(data, general_fraud_model)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Ensure the data is in the correct format
    input_df = pd.DataFrame([data])  # Wrap data in a list to ensure it is treated as a single row

    # Make predictions using the general fraud model
    try:
        predictions = general_fraud_model.predict(input_df)
        logger.info("Prediction made using General Fraud Model.")
    except Exception as e:
        logger.error(f"Error making prediction with General Fraud Model: {e}")
        return jsonify({'error': str(e)}), 500

    # Return predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
