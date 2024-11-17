import sys
import os
from flask import Flask
import logging

# Add the fraud_detection root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Go two levels up
sys.path.append(project_root)

from fraud_detection.api.endpoints import predict_credit_fraud, predict_general_fraud

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
