# fraud_detection/api/flask_app.py
from flask import Flask
from fraud_detection.api.endpoints import predict_credit_fraud, predict_general_fraud
import logging

# Initialize the Flask application
app = Flask(__name__)

# Set up logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the routes for predictions
app.add_url_rule('/predict/credit_fraud', 'predict_credit_fraud', predict_credit_fraud, methods=['POST'])
app.add_url_rule('/predict/general_fraud', 'predict_general_fraud', predict_general_fraud, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)