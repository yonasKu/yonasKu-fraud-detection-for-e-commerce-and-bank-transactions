# tests/test_api.py

import unittest
import json
from flask import Flask
from src.api import api

# Initialize Flask app and register the API
app = Flask(__name__)
app.register_blueprint(api)

# Function to get expected features based on the model type
def get_expected_features(model_type):
    if model_type == 'credit_card':
        return {
            'Time': 'float',
            'V1': 'float',
            'V2': 'float',
            'V3': 'float',
            'V4': 'float',
            'V5': 'float',
            'V6': 'float',
            'V7': 'float',
            'V8': 'float',
            'V9': 'float',
            'V10': 'float',
            'V11': 'float',
            'V12': 'float',
            'V13': 'float',
            'V14': 'float',
            'V15': 'float',
            'V16': 'float',
            'V17': 'float',
            'V18': 'float',
            'V19': 'float',
            'V20': 'float',
            'V21': 'float',
            'V22': 'float',
            'V23': 'float',
            'V24': 'float',
            'V25': 'float',
            'V26': 'float',
            'V27': 'float',
            'V28': 'float',
            'Amount': 'float'
        }
    
    elif model_type == 'fraud':
        return {
            'user_id': 'str',
            'signup_time': 'datetime',  # Assume ISO 8601 format
            'purchase_time': 'datetime',
            'purchase_value': 'float',
            'device_id': 'str',
            'source': 'str',
            'browser': 'str',
            'sex': 'str',
            'age': 'int',
            'ip_address': 'str',
            'country': 'str',
            'transaction_frequency': 'int',
            'transaction_velocity': 'int',
            'hour_of_day': 'int',
            'day_of_week': 'int'
        }
    
    else:
        raise ValueError("Unknown model type. Please specify 'credit_card' or 'fraud'.")

class APITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app.test_client()
        cls.app.testing = True

    def test_predict_credit_card(self):
        # Retrieve expected features
        expected_features = get_expected_features('credit_card')
        
        # Create a sample input with the expected types
        sample_data = {key: (0.0 if dtype == 'float' else
                             100 if dtype == 'float' and key == 'Amount' else
                             0 if dtype == 'int' else
                             'string' if dtype == 'str' else
                             '2023-01-01T12:00:00Z' if dtype == 'datetime' else None)
                        for key, dtype in expected_features.items()}
        
        # Make a POST request to the credit card prediction endpoint
        response = self.app.post('/api/predict/credit_card', data=json.dumps(sample_data),
                                 content_type='application/json')
        
        # Check if the response status code is 200
        self.assertEqual(response.status_code, 200)

        # Verify the prediction and probability are in the response
        response_json = json.loads(response.data)
        self.assertIn('prediction', response_json)
        self.assertIn('fraud_probability', response_json)

    def test_predict_fraud(self):
        # Retrieve expected features
        expected_features = get_expected_features('fraud')
        
        # Create a sample input with the expected types
        sample_data = {key: (0 if dtype == 'int' else
                             150.0 if dtype == 'float' and key == 'purchase_value' else
                             'string' if dtype == 'str' else
                             '2023-01-01T12:00:00Z' if dtype == 'datetime' else None)
                        for key, dtype in expected_features.items()}
        
        # Make a POST request to the fraud prediction endpoint
        response = self.app.post('/api/predict/fraud', data=json.dumps(sample_data),
                                 content_type='application/json')
        
        # Check if the response status code is 200
        self.assertEqual(response.status_code, 200)

        # Verify the prediction and probability are in the response
        response_json = json.loads(response.data)
        self.assertIn('prediction', response_json)
        self.assertIn('fraud_probability', response_json)

# Entry point for running the tests
if __name__ == '__main__':
    unittest.main()
