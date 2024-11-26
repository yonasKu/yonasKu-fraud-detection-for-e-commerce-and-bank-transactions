import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../fraud_detection')))

from fraud_detection.model_training import (
    train_and_save_model,
    train_logistic_regression,
    train_random_forest,
)

from fraud_detection.model_training import prepare_credit_card_data
import joblib

class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up shared test data and configurations."""
        cls.data_path = "data/creditcard.csv"  # Ensure this path is correct
        cls.target_column = "Class"
        cls.test_size = 0.2
        cls.scaler_path = "models/scaler.pkl"
        cls.model_path = "models/test_model.pkl"

        # Prepare data
        cls.data = prepare_credit_card_data(
            cls.data_path,
            cls.target_column,
            test_size=cls.test_size,
            scale_data=True,
            scaler_path=cls.scaler_path,
        )

    def test_logistic_regression_training(self):
        """Test training logistic regression model."""
        results = train_logistic_regression(
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_train"],
            self.data["y_test"],
            self.model_path,
            scaler=self.data["scaler"],
            scaler_path=self.scaler_path,
        )
        self.assertIn("roc_auc", results)
        self.assertGreater(results["roc_auc"], 0.5)
        self.assertTrue(os.path.exists(self.model_path))

    def test_random_forest_training(self):
        """Test training random forest model."""
        results = train_random_forest(
            self.data["X_train"],
            self.data["X_test"],
            self.data["y_train"],
            self.data["y_test"],
            self.model_path,
            scaler=self.data["scaler"],
            scaler_path=self.scaler_path,
        )
        self.assertIn("roc_auc", results)
        self.assertGreater(results["roc_auc"], 0.5)
        self.assertTrue(os.path.exists(self.model_path))

    def test_model_saving_and_loading(self):
        """Test that the trained model can be saved and loaded."""
        model = joblib.load(self.model_path)
        self.assertIsInstance(model, (LogisticRegression, RandomForestClassifier))

    @classmethod
    def tearDownClass(cls):
        """Clean up files after tests."""
        if os.path.exists(cls.scaler_path):
            os.remove(cls.scaler_path)
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)


if __name__ == "__main__":
    unittest.main()
