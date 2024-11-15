# src/serve_model.py

# src/serve_model.py
import joblib

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """Load the trained model from the specified path."""
        model = joblib.load(self.model_path)
        return model

# Instantiate the model loaders
credit_card_model_loader = ModelLoader('model/Credit Card Fraud Detection with Random Forest_random_forest_model.pkl')
fraud_model_loader = ModelLoader('model/Fraud Detection with Random Forest_random_forest_model.pkl')
