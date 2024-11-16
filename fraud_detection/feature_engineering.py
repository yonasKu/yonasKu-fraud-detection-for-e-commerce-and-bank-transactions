import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
def preprocess_transaction_data(fraud_data_df):
    """
    Preprocess the fraud data by calculating transaction-related features.
    
    Args:
        fraud_data_df (pd.DataFrame): The fraud data DataFrame.
        
    Returns:
        pd.DataFrame: The fraud data with added transaction-related features.
    """
    # Convert `purchase_time` and `signup_time` to datetime objects for easier manipulation
    fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])
    fraud_data_df['signup_time'] = pd.to_datetime(fraud_data_df['signup_time'])

    # Step 1: Transaction Frequency per User (same as the number of purchases per user)
    fraud_data_df['transaction_frequency'] = fraud_data_df.groupby('user_id')['purchase_time'].transform('count')

    # Step 2: Transaction Count (just counts the number of transactions per user)
    fraud_data_df['transaction_count'] = fraud_data_df.groupby('user_id')['user_id'].transform('count')

    # Step 3: Transaction Velocity (time difference between signup and purchase)
    fraud_data_df['transaction_velocity'] = (fraud_data_df['purchase_time'] - fraud_data_df['signup_time']).dt.total_seconds()

    # Step 4: Sort data by `user_id` and `purchase_time` to calculate purchase difference
    fraud_data_df = fraud_data_df.sort_values(by=['user_id', 'purchase_time'])

    # # Step 5: Purchase Difference (time difference between consecutive purchases for the same user)
    # fraud_data_df['purchase_diff'] = fraud_data_df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()

    # Step 6: Extract Hour of the Day and Day of the Week from `purchase_time`
    fraud_data_df['hour_of_day'] = fraud_data_df['purchase_time'].dt.hour
    fraud_data_df['day_of_week'] = fraud_data_df['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6

    return fraud_data_df

def label_encode_categorical_columns(fraud_data_df):
    """
    Label encodes categorical columns in the fraud_data_df DataFrame.
    
    Args:
        fraud_data_df (pd.DataFrame): The DataFrame containing the fraud data.
        
    Returns:
        pd.DataFrame: The DataFrame with label encoded categorical columns.
    """
    # Identify categorical columns (object or category types)
    categorical_columns = fraud_data_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Initialize LabelEncoder
    label_encoders = {}

    # Apply Label Encoding for each categorical column
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        fraud_data_df[col] = label_encoders[col].fit_transform(fraud_data_df[col])

    return fraud_data_df, label_encoders

def standardize_numerical_features(fraud_data_df, exclude_columns):
    """
    Standardizes the numerical features in the DataFrame, excluding specific columns.
    
    Args:
        fraud_data_df (pd.DataFrame): The DataFrame containing the fraud data.
        exclude_columns (list): List of columns to exclude from scaling.
        
    Returns:
        pd.DataFrame: The DataFrame with standardized numerical features.
    """
    # Identify numerical columns (excluding excluded ones)
    numerical_columns = fraud_data_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col not in exclude_columns]

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Standardize the numerical features
    fraud_data_df[numerical_columns] = scaler.fit_transform(fraud_data_df[numerical_columns])

    return fraud_data_df