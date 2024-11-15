import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import sparse

# Function to load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to display the overview of the dataset
def data_overview(df):
    print("Data Overview:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Function to standardize numerical features
from sklearn.preprocessing import StandardScaler

def standardize_numerical_features(df, exclude_columns=None):
    """
    Standardize numerical features in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing numerical features to be standardized.
    - exclude_columns (list): List of columns to exclude from scaling (e.g., categorical or label columns).
    
    Returns:
    - df (pd.DataFrame): DataFrame with standardized numerical features.
    """
    # Initialize the scaler
    scaler = StandardScaler()

    # Identify numerical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude specific columns (like device_id, browser, class, etc.)
    if exclude_columns:
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

    # Check if there are any numerical columns to scale
    if numeric_columns:
        # Standardize the numerical features
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df


def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing categorical features to be encoded.
    
    Returns:
    - df (pd.DataFrame or sparse DataFrame): DataFrame with one-hot encoded categorical features.
    """
    # Check if the DataFrame is empty
    if df.empty:
        return df
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encode categorical features
    if categorical_columns:
        # Use sparse=True to create a sparse DataFrame
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, sparse=True)

    return df