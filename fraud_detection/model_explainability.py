import shap
from utils.logging_util import setup_logger
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
# Setup logger
logger = setup_logger('../logs/explainability.log')



def explain_with_shap(model, X_train, X_test, feature_names=None, sample_size=100):
    """
    Generate and visualize SHAP explanations for the given model.

    Parameters:
        - model: Trained model to explain.
        - X_train: Training data used to fit the SHAP explainer.
        - X_test: Test data to analyze and visualize.
        - feature_names: List of feature names (optional).
        - sample_size: Number of test samples to explain.

    Returns:
        - shap_values: Computed SHAP values.
        - shap_summary: SHAP summary visualization.
    """
    try:
        logger.info("Initializing SHAP explainer...")
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model, X_train, feature_names=feature_names)
        else:
            explainer = shap.Explainer(model.predict, X_train)

        # Compute SHAP values
        logger.info("Computing SHAP values...")
        X_sample = X_test[:sample_size]
        shap_values = explainer(X_sample)

        # Plot summary
        logger.info("Generating SHAP summary plot...")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names)

        return shap_values

    except Exception as e:
        logger.error(f"Error in SHAP explanation: {e}")
        raise


def explain_with_lime(model, X_train, X_test, y_train, feature_names, sample_index=0, num_features=10, class_names=None):
    """
    Generate and visualize LIME explanations for the given model.

    Parameters:
        - model: Trained model to explain.
        - X_train: Training data for LIME explainer.
        - X_test: Test data to analyze and visualize.
        - y_train: Target variable from training data (for scaling).
        - feature_names: List of feature names.
        - sample_index: Index of the test sample to explain.
        - num_features: Number of top features to display in the explanation.
        - class_names: Names of the output classes (for classification).

    Returns:
        - explanation: LIME explanation object.
    """
    try:
        logger.info("Initializing LIME explainer...")
        lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification" if hasattr(model, "predict_proba") else "regression",
            discretize_continuous=True,
        )

        # Select the sample to explain
        sample = X_test[sample_index]
        logger.info(f"Explaining sample index {sample_index}...")

        # Generate explanation
        explanation = lime_explainer.explain_instance(
            sample, model.predict_proba if hasattr(model, "predict_proba") else model.predict, num_features=num_features
        )

        # Visualize explanation
        logger.info("Generating LIME visualization...")
        explanation.show_in_notebook(show_table=True)
        explanation.as_pyplot_figure()
        plt.show()

        return explanation

    except Exception as e:
        logger.error(f"Error in LIME explanation: {e}")
        raise


def explain_model_with_shap_and_lime(
    model, X_train, X_test, y_train, feature_names, sample_size=100, lime_sample_index=0, num_lime_features=10, class_names=None
):
    """
    Combined explanation using both SHAP and LIME.

    Parameters:
        - model: Trained model to explain.
        - X_train, X_test: Training and test datasets.
        - y_train: Training target data (for LIME scaling).
        - feature_names: List of feature names.
        - sample_size: Number of samples for SHAP.
        - lime_sample_index: Index of the sample for LIME.
        - num_lime_features: Number of LIME features to explain.
        - class_names: Names of classes for LIME.

    Returns:
        - shap_values: SHAP values.
        - lime_explanation: LIME explanation object.
    """
    logger.info("Starting model explanation using SHAP and LIME...")
    shap_values = explain_with_shap(model, X_train, X_test, feature_names, sample_size)
    lime_explanation = explain_with_lime(
        model, X_train, X_test, y_train, feature_names, lime_sample_index, num_lime_features, class_names
    )
    logger.info("Explanation complete.")
    return shap_values, lime_explanation