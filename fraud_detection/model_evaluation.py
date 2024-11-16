from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.logging_util import setup_logger

# Setup logger
logger = setup_logger('../logs/training.log')

def evaluate_model(model, X_test, y_test):
    """Evaluate a model using multiple metrics."""
    try:
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC AUC if the model supports it
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # get probability for class 1
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log evaluation metrics
        logger.info(f"Evaluation metrics for {model.__class__.__name__}:")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"ROC AUC: {roc_auc}")
        logger.info(f"Confusion Matrix: \n{conf_matrix}")

        # Return all metrics as a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
