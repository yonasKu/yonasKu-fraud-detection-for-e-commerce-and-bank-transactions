import sys
import os
from flask import Flask
import logging

# Add the fraud_detection folder to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Going up to the root directory
sys.path.append(project_root)

# Import the endpoints and the stats blueprint
from fraud_detection.api.endpoints import predict_credit_fraud, predict_general_fraud, stats_blueprint

# Initialize the Flask application
app = Flask(__name__)

# Set up logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register the blueprint for the statistics endpoints
app.register_blueprint(stats_blueprint, url_prefix='/stats')  # Optional URL prefix: '/stats'

# Define the routes for predictions
app.add_url_rule('/predict/credit_fraud', 'predict_credit_fraud', predict_credit_fraud, methods=['POST'])
app.add_url_rule('/predict/general_fraud', 'predict_general_fraud', predict_general_fraud, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
