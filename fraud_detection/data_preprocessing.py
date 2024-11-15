import pandas as pd


from utils.logging_util import setup_logger

# Setup logger
logger = setup_logger('../logs/preprocessing.log')

def preprocess_data(creditcard_df, fraud_data_df, ip_to_country_df):
    """
    Preprocess datasets by handling missing values, duplicates, and data type conversion.

    Args:
        creditcard_df (pd.DataFrame): Credit card transactions dataset.
        fraud_data_df (pd.DataFrame): E-commerce fraud dataset.
        ip_to_country_df (pd.DataFrame): IP to country mapping dataset.

    Returns:
        tuple: Cleaned DataFrames for credit card, fraud, and IP-country datasets.
    """
    logger.info("Starting data preprocessing...")

    # Handle missing values
    logger.info("Dropping missing values...")
    creditcard_df.dropna(inplace=True)
    fraud_data_df.dropna(inplace=True)
    ip_to_country_df.dropna(inplace=True)

    # Remove duplicates
    logger.info("Dropping duplicates...")
    creditcard_df.drop_duplicates(inplace=True)
    fraud_data_df.drop_duplicates(inplace=True)
    ip_to_country_df.drop_duplicates(inplace=True)

    # Convert data types where appropriate
    logger.info("Converting data types...")
    fraud_data_df['signup_time'] = pd.to_datetime(fraud_data_df['signup_time'])
    fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])

    logger.info("Data preprocessing complete.")
    return creditcard_df, fraud_data_df, ip_to_country_df
