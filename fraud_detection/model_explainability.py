import shap
from utils.logging_util import setup_logger
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Setup logger
logger = setup_logger('../logs/explainability.log')

def explain_model_shap(model, X_train, X_test, model_type='rf'):
    """Generate SHAP explanations for the trained model."""
    try:
        logger.info(f"Starting explanation for model type: {model_type}.")
        
        # Choose the appropriate SHAP explainer based on model type
        if model_type == 'rf':
            explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer for Random Forest model.")
        elif model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer for XGBoost model.")
        elif model_type == 'lr':
            explainer = shap.KernelExplainer(model.predict_proba, X_train)
            logger.info("Using KernelExplainer for Logistic Regression model.")
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        # Generate SHAP values for the training dataset
        shap_values = explainer.shap_values(X_train)
        logger.info("SHAP values computed.")

        # Visualize SHAP plots (these will show up interactively)
        logger.info("Generating SHAP summary plot...")
        
        # SHAP summary plot (Feature Importance)
        shap.summary_plot(shap_values, X_train, plot_type="bar")  # Feature importance plot
        shap.summary_plot(shap_values, X_train)  # Detailed SHAP value distribution plot

        # SHAP Force plot for a specific instance (e.g., first instance in the test set)
        logger.info("Generating SHAP force plot...")
        shap.initjs()  # Initialize JS for force plot visualization
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_train.iloc[0], matplotlib=True)  # Force plot for the first instance
        
        # SHAP dependence plot (for a specific feature)
        logger.info("Generating SHAP dependence plot...")
        feature_name = X_train.columns[0]  # You can change this to a specific feature
        shap.dependence_plot(feature_name, shap_values, X_train)  # Replace "feature_name" with an actual feature
        
        logger.info("SHAP explanation completed successfully.")
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
