# fraud_detection/model_training.py
import time
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.optimizers import Adam
from utils.logging_util import setup_logger

# Setup logger
logger = setup_logger('../logs/training.log')


def prepare_data(data_path, target_column, test_size=0.2, scale_data=True):
    """Prepare data for training and testing, handling timestamps and categorical variables."""
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")

        # Convert timestamp columns to datetime
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Calculate time difference between signup and purchase (in seconds)
        df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

        # Drop the original timestamp columns
        df.drop(columns=['signup_time', 'purchase_time'], inplace=True)

        # Handle categorical columns by encoding them
        categorical_columns = ['device_id', 'source', 'browser', 'sex']
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            logger.info(f"Encoded column: {col}")

        # Define features (X) and target (y)
        X = df.drop(columns=[target_column])  # Features (everything except the target column)
        y = df[target_column]  # Target

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info("Data successfully split into train and test sets.")

        # Scale the data if needed
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info("Standardized the data.")

        # Return the prepared data
        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def train_and_save_model(model, X_train, X_test, y_train, y_test, model_path):
    """Train and save a model to disk."""
    try:
        logger.info(f"Training model: {model.__class__.__name__}.")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Save the model to disk
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None

        report = classification_report(y_test, y_pred)
        logger.info("Model training and evaluation completed successfully.")
        return {"model": model, "training_time": training_time, "roc_auc": roc_auc, "report": report}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


# Individual training functions
def train_logistic_regression(X_train, X_test, y_train, y_test, model_path):
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_decision_tree(X_train, X_test, y_train, y_test, model_path):
    model = DecisionTreeClassifier()
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_random_forest(X_train, X_test, y_train, y_test, model_path):
    model = RandomForestClassifier(class_weight="balanced")
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_gradient_boosting(X_train, X_test, y_train, y_test, model_path):
    model = GradientBoostingClassifier(n_estimators=100)
    return train_and_save_model(model, X_train, X_test, y_train, y_test, model_path)


def train_mlp(X_train, X_test, y_train, y_test, model_path):
    """Train a Multi-Layer Perceptron (MLP) model and save it to disk."""
    try:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=20, batch_size=32, verbose=2)

        model.save(model_path)
        logger.info(f"MLP model saved to {model_path}.")
        return model, history.history
    except Exception as e:
        logger.error(f"Error training MLP model: {e}")
        raise


def train_lstm(X_train, X_test, y_train, y_test, model_path):
    """Train a Long Short-Term Memory (LSTM) model and save it to disk."""
    try:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(1, X_train.shape[2])),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=20, batch_size=32, verbose=2)

        model.save(model_path)
        logger.info(f"LSTM model saved to {model_path}.")
        return model, history.history
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        raise


def train_cnn(X_train, X_test, y_train, y_test, model_path):
    """Train a Convolutional Neural Network (CNN) model and save it to disk."""
    try:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=20, batch_size=32, verbose=2)

        model.save(model_path)
        logger.info(f"CNN model saved to {model_path}.")
        return model, history.history
    except Exception as e:
        logger.error(f"Error training CNN model: {e}")
        raise


def train_rnn(X_train, X_test, y_train, y_test, model_path):
    """Train a Recurrent Neural Network (RNN) model and save it to disk."""
    try:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        model = Sequential([
            SimpleRNN(64, activation='tanh', input_shape=(1, X_train.shape[2])),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=20, batch_size=32, verbose=2)

        model.save(model_path)
        logger.info(f"RNN model saved to {model_path}.")
        return model, history.history
    except Exception as e:
        logger.error(f"Error training RNN model: {e}")
        raise
