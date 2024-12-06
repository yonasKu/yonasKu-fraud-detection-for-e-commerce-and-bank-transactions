import shap
from utils.logging_util import setup_logger
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Setup logger
logger = setup_logger('../logs/explainability.log')

def explain_model_shap(model, X_train, X_test, model_type='rf', sample_size=100):
    """Generate SHAP explanations with memory optimization."""
    try:
        logger.info(f"Starting explanation for model type: {model_type}.")
        
        # Reduce the dataset size
        X_train_sampled = X_train.sample(n=sample_size, random_state=42)
        logger.info(f"Using a sample size of {sample_size} for SHAP.")

        # Choose SHAP explainer
        if model_type in ['rf', 'xgboost']:
            explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer.")
        elif model_type == 'lr':
            # Reduce the background data size using shap.sample
            background_data = shap.sample(X_train_sampled, 1000)  # Use 1000 samples instead of the full set
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            logger.info("Using KernelExplainer.")
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        # Compute SHAP values for a small test subset
        X_test_sampled = X_test.sample(n=10, random_state=42)  # Explain 10 test instances
        shap_values = explainer.shap_values(X_test_sampled)
        logger.info("SHAP values computed.")

        # Visualizations
        shap.summary_plot(shap_values, X_train_sampled, plot_type="bar", max_display=10)  # Top 10 features
        shap.summary_plot(shap_values, X_train_sampled, max_display=10)

        return shap_values

    except Exception as e:
        logger.error(f"Error in explaining model: {e}")
        raise


def explain_model_lime(model, X_train, X_test, idx=0):
    """Explain a specific instance using LIME."""
    try:
        # Create a LIME explainer object
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=['Non-Fraud', 'Fraud'],  # Adjust for your classes
            mode='classification'
        )

        # Select a data point to explain
        instance = X_test.iloc[idx]
        logger.info(f"Explaining instance {idx}...")

        # Explain the prediction
        explanation = explainer.explain_instance(instance.values, model.predict_proba, num_features=5)
        
        # Display the explanation in notebook (if using Jupyter)
        explanation.show_in_notebook(show_table=True, show_all=False)  # In Jupyter Notebook
        explanation.as_pyplot_figure()
        plt.show()

        return explanation

    except Exception as e:
        logger.error(f"Error in LIME explanation: {e}")
        raise
