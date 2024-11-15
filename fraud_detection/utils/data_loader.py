import pandas as pd

def load_datasets(creditcard_path, fraud_data_path, ip_to_country_path):
    """
    Load datasets into pandas DataFrames.
    
    Args:
        creditcard_path (str): Path to the credit card transactions file.
        fraud_data_path (str): Path to the e-commerce fraud data file.
        ip_to_country_path (str): Path to the IP to country mapping file.
    
    Returns:
        tuple: DataFrames for credit card, fraud, and IP-country datasets.
    """
    creditcard_df = pd.read_csv(creditcard_path)
    fraud_data_df = pd.read_csv(fraud_data_path)
    ip_to_country_df = pd.read_csv(ip_to_country_path)

    return creditcard_df, fraud_data_df, ip_to_country_df
