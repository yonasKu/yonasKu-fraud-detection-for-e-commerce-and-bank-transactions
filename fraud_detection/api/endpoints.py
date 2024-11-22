# fraud_detection/api/endpoints.py

import joblib
import pandas as pd
import logging
from flask import jsonify, request, Blueprint
import os
import numpy as np

# Initialize logging for the endpoints
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the credit fraud detection model
def load_credit_fraud_model():
    # Corrected model path based on folder structure
    credit_fraud_model_path = os.path.join('models', 'credit_gradient_boosting_model.pkl')
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
    general_fraud_model_path = os.path.join('models', 'gradient_boosting_model.pkl')
    try:
        general_fraud_model = joblib.load(general_fraud_model_path)
        logger.info("General Fraud Model loaded successfully.")
        return general_fraud_model
    except Exception as e:
        logger.error(f"Error loading General Fraud Model: {e}")
        return None

# Define the specific feature sets for each model
# general_fraud_model_features = [
#     'user_id', 'purchase_value', 'device_id', 'source', 
#     'browser', 'sex', 'age', 'country', 
#     'transaction_frequency', 'transaction_velocity', 
#     'hour_of_day', 'day_of_week'
# ]
general_fraud_model_features = ['purchase_value', 'age', 'transaction_frequency', 'transaction_count', 
                    'transaction_velocity', 'hour_of_day', 'day_of_week', 'time_to_purchase',
                    'country_encoded', 'source_Direct', 'source_SEO', 'browser_FireFox', 
                    'browser_IE', 'browser_Opera', 'browser_Safari', 'sex_M']

credit_fraud_model_features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']

# Helper function to ensure input data has required columns
def validate_input(data, model_type):
    # Set the required columns based on the model type
    if model_type == 'credit_fraud':
        required_columns = credit_fraud_model_features
    elif model_type == 'general_fraud':
        required_columns = general_fraud_model_features
    else:
        return False, "Invalid model type."
    
    # Check if all required columns are present in the data
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
    is_valid, error_msg = validate_input(data, 'credit_fraud')
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
    is_valid, error_msg = validate_input(data, 'general_fraud')
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


# Blueprint for statistics
stats_blueprint = Blueprint('stats', __name__)

# fraud_data_path = os.path.join('data', 'Fraud_Data.csv')
fraud_data_path = os.path.join('data', 'Fraudlent_Datas.csv')


# Helper function to convert numpy types to native Python types
def convert_data_types(data):
    if isinstance(data, np.ndarray):  # If it's a numpy array
        return data.tolist()  # Convert numpy array to a list
    elif isinstance(data, pd.DataFrame):  # If it's a pandas DataFrame
        # Convert all columns of the dataframe to native Python types
        return data.applymap(lambda x: x.item() if isinstance(x, np.generic) else x).to_dict(orient='records')
    elif isinstance(data, dict):  # If it's a dictionary
        return {key: convert_data_types(value) for key, value in data.items()}
    elif isinstance(data, list):  # If it's a list
        return [convert_data_types(item) for item in data]
    elif isinstance(data, np.generic):  # If it's a numpy scalar
        return data.item()  # Convert numpy scalar to native Python type (int, float)
    else:
        return data  # If it's already a native Python type, return as-is

# Load the fraud data
def load_fraud_data():
    try:
        data = pd.read_csv(fraud_data_path, parse_dates=['signup_time', 'purchase_time'])
        return data, None  # Return the data and a None error
    except Exception as e:
        return None, str(e)


@stats_blueprint.route('/general_summary', methods=['GET'])
def get_general_summary():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    total_users = len(data)
    total_fraud_cases = data['class'].sum()
    fraud_rate = (total_fraud_cases / total_users) * 100
    average_purchase_value = data['purchase_value'].mean()

    summary = {
        'total_users': total_users,
        'total_fraud_cases': total_fraud_cases,
        'fraud_rate': fraud_rate,
        'average_purchase_value': average_purchase_value
    }

    # Convert summary data to JSON serializable format
    summary = convert_data_types(summary)

    return jsonify(summary)

@stats_blueprint.route('/age_distribution', methods=['GET'])
def get_age_distribution():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Group by age and sum fraud cases (class = 1 indicates fraud)
    age_distribution = data.groupby('age')['class'].sum().reset_index()

    # Convert age distribution data to JSON serializable format
    age_distribution = convert_data_types(age_distribution)

    return jsonify(age_distribution)

# Endpoint for sex-based fraud analysis
@stats_blueprint.route('/sex_distribution', methods=['GET'])
def get_sex_distribution():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Group by sex and sum fraud cases (class = 1 indicates fraud)
    sex_distribution = data.groupby('sex')['class'].sum().reset_index()

    # Convert sex distribution data to JSON serializable format
    sex_distribution = convert_data_types(sex_distribution)

    return jsonify(sex_distribution)
# Endpoint for top purchases by users
@stats_blueprint.route('/top_purchases', methods=['GET'])
def get_top_purchases():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Sort by purchase value and get the top 10 purchases
    top_purchases = data[['user_id', 'purchase_value']].sort_values(by='purchase_value', ascending=False).head(10)

    # Convert top purchases data to JSON serializable format
    top_purchases = convert_data_types(top_purchases)

    return jsonify(top_purchases)



# Endpoint for fraud rate by source
@stats_blueprint.route('/fraud_by_source', methods=['GET'])
def get_fraud_by_source():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Group by source and calculate fraud rate
    fraud_by_source = data.groupby('source')['class'].mean().reset_index()

    # Convert fraud by source data to JSON serializable format
    fraud_by_source = convert_data_types(fraud_by_source)

    return jsonify(fraud_by_source)

# Example stats endpoint (for testing conversion)

@stats_blueprint.route('/summary', methods=['GET'])
def summary_stats():
    # Example stats data, possibly containing numpy.int64
    data = {
        'total_transactions': np.int64(1000),
        'fraud_transactions': np.int64(200),
        'total_revenue': np.float64(50000.50),
    }
    
    # Convert data types to JSON serializable formats
    data = convert_data_types(data)
    
    return jsonify(data)

# Endpoint for fraud trends (line chart)
@stats_blueprint.route('/trends', methods=['GET'])
def get_fraud_trends():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Group by purchase date and count fraudulent transactions
    data['purchase_date'] = data['purchase_time'].dt.date
    trends = data.groupby('purchase_date')['class'].sum().reset_index()

    # Convert trends data to JSON serializable format
    trends = convert_data_types(trends)

    return jsonify(trends)

# Endpoint for geographic analysis (only IP addresses and fraud classification)
@stats_blueprint.route('/geography', methods=['GET'])
def get_geographic_fraud():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Just return the unique IP addresses and fraud classification (class) information
    # ip_data = data[['ip_address', 'class']].drop_duplicates()
    ip_data = data[['ip_address', 'class', 'country']].drop_duplicates()
    # Convert IP data to JSON serializable format
    ip_data = convert_data_types(ip_data)

    return jsonify(ip_data)


@stats_blueprint.route('/devices_fraud', methods=['GET'])
def get_device_fraud():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Filter the data to only include fraudulent users (class = 1)
    fraudulent_data = data[data['class'] == 1]

    # Group by device_id and browser, and count fraudulent transactions
    device_fraud = fraudulent_data.groupby(['device_id', 'browser'])['class'].sum().reset_index()

    # Convert device fraud data to JSON serializable format
    device_fraud = convert_data_types(device_fraud)

    return jsonify(device_fraud)

# You can also modify the `/devices` endpoint name if you want to keep both routes:
@stats_blueprint.route('/devices', methods=['GET'])
def get_device_stats():
    data, error = load_fraud_data()
    if error:
        return jsonify({'error': f"Failed to load data: {error}"}), 500

    # Group by device_id and browser, and count fraudulent transactions
    devices = data.groupby(['device_id', 'browser'])['class'].sum().reset_index()

    # Convert device data to JSON serializable format
    devices = convert_data_types(devices)

    return jsonify(devices)