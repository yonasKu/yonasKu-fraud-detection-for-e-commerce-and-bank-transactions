import unittest
import pandas as pd
import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_detection')))

from fraud_detection.data_preprocessing import preprocess_data, int_to_ip, ip_to_int

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up dummy data for testing."""
        self.creditcard_df = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'Amount': [100.0, 150.0, 200.0],
            'Fraud': [0, 1, 0]
        })
        self.fraud_data_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'signup_time': ['2021-01-01', '2021-02-01', '2021-03-01'],
            'purchase_time': ['2021-01-02', '2021-02-02', '2021-03-02'],
            'purchase_value': [300.0, 450.0, 500.0],
            'ip_address': [3232235777, 3232235778, 3232235779],
        })
        self.ip_to_country_df = pd.DataFrame({
            'lower_bound_ip_address': [3232235776, 3232235778],
            'upper_bound_ip_address': [3232235777, 3232235779],
            'country': ['US', 'CA']
        })

    def test_preprocess_data(self):
        """Test the preprocess_data function."""
        cleaned_creditcard_df, cleaned_fraud_data_df, cleaned_ip_to_country_df = preprocess_data(
            self.creditcard_df,
            self.fraud_data_df,
            self.ip_to_country_df
        )

        # Check if missing values are handled
        self.assertFalse(cleaned_creditcard_df.isnull().values.any())
        self.assertFalse(cleaned_fraud_data_df.isnull().values.any())
        self.assertFalse(cleaned_ip_to_country_df.isnull().values.any())

        # Check if duplicates are removed
        self.assertEqual(len(cleaned_creditcard_df), len(self.creditcard_df))
        self.assertEqual(len(cleaned_fraud_data_df), len(self.fraud_data_df))
        self.assertEqual(len(cleaned_ip_to_country_df), len(self.ip_to_country_df))

    def test_int_to_ip(self):
        """Test the int_to_ip function."""
        ip = int_to_ip(3232235777)
        self.assertEqual(ip, '192.168.1.1')

    def test_ip_to_int(self):
        """Test the ip_to_int function."""
        ip_int = ip_to_int('192.168.1.1')
        self.assertEqual(ip_int, 3232235777)

if __name__ == '__main__':
    unittest.main()
