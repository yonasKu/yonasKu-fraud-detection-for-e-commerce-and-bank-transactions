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

def int_to_ip(ip_int):
    """
    Converts an integer IP address to the dot-decimal format (e.g., '192.168.0.1').
    """
    if not isinstance(ip_int, int):
        raise ValueError(f"Expected an integer, got {type(ip_int).__name__}")
    return '.'.join([str((ip_int >> (i * 8)) & 0xFF) for i in range(3, -1, -1)])

# Function to convert an IP address from dot-decimal format to integer
def ip_to_int(ip_str):
    """
    Converts a dot-decimal IP address (e.g., '192.168.0.1') to its integer equivalent.
    """
    octets = ip_str.split('.')
    return (int(octets[0]) << 24) + (int(octets[1]) << 16) + (int(octets[2]) << 8) + int(octets[3])



def int_to_ip(ip_int):
    """
    Converts an integer IP address to the dot-decimal format (e.g., '192.168.0.1').
    """
    if not isinstance(ip_int, int):
        raise ValueError(f"Expected an integer, got {type(ip_int).__name__}")
    return '.'.join([str((ip_int >> (i * 8)) & 0xFF) for i in range(3, -1, -1)])


def ip_to_int(ip_str):
    """
    Converts a dot-decimal IP address (e.g., '192.168.0.1') to its integer equivalent.
    """
    octets = ip_str.split('.')
    return (int(octets[0]) << 24) + (int(octets[1]) << 16) + (int(octets[2]) << 8) + int(octets[3])


def preprocess_ip_data(fraud_data_df, ipaddress_to_country_df):
    """
    Preprocesses the fraud data and IP address to country data by converting IP addresses 
    to integers and merging the datasets based on IP ranges.
    """
    logger.info("Starting IP preprocessing...")

    # Step 1: Ensure IP columns are treated as integers for merging
    fraud_data_df['ip_address'] = fraud_data_df['ip_address'].fillna(0).astype(int)
    ipaddress_to_country_df['lower_bound_ip_address'] = ipaddress_to_country_df['lower_bound_ip_address'].fillna(0).astype(int)
    ipaddress_to_country_df['upper_bound_ip_address'] = ipaddress_to_country_df['upper_bound_ip_address'].fillna(0).astype(int)

    # Step 2: Convert IP addresses to dot-decimal format (for display purposes only)
    ipaddress_to_country_df['lower_bound_ip_address_display'] = ipaddress_to_country_df['lower_bound_ip_address'].apply(int_to_ip)
    ipaddress_to_country_df['upper_bound_ip_address_display'] = ipaddress_to_country_df['upper_bound_ip_address'].apply(int_to_ip)
    fraud_data_df['ip_address_display'] = fraud_data_df['ip_address'].apply(int_to_ip)

    # Step 3: Merge the datasets based on IP address ranges
    fraud_data_df = pd.merge_asof(
        fraud_data_df.sort_values('ip_address'),
        ipaddress_to_country_df.sort_values('lower_bound_ip_address'),
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'  # This ensures we find the closest lower_bound_ip_address <= ip_address
    )

    # Step 4: Filter rows where ip_address falls between lower_bound_ip_address and upper_bound_ip_address
    fraud_data_df = fraud_data_df[
        (fraud_data_df['ip_address'] >= fraud_data_df['lower_bound_ip_address']) & 
        (fraud_data_df['ip_address'] <= fraud_data_df['upper_bound_ip_address'])
    ]

    # Step 5: Drop unnecessary columns (if needed)
    fraud_data_df = fraud_data_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address', 
                                                'lower_bound_ip_address_display', 'upper_bound_ip_address_display'])

    # Step 6: Ensure the 'country' column is filled properly
    fraud_data_df['country'] = fraud_data_df['country'].fillna('Unknown')

    # Final dataset should have the required columns
    fraud_data_df = fraud_data_df[['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                                   'device_id', 'source', 'browser', 'sex', 'age', 
                                   'ip_address', 'class', 'country']]

    logger.info("IP data preprocessing complete.")
    return fraud_data_df