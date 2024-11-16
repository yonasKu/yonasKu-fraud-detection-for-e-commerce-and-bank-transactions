import matplotlib.pyplot as plt
import seaborn as sns
from utils.logging_util import setup_logger

# Set up the logger
logger = setup_logger('../logs/eda_fraud_detection.log')  # Specify path for the log file

def plot_transaction_amount_distribution(df):
    """ Plot distribution of transaction amounts for credit card data. """
    logger.info("Starting plot: Transaction Amount Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Distribution of Transaction Amounts (Credit Card Data)')
    plt.show()
    logger.info("Completed plot: Transaction Amount Distribution")

def plot_fraudulent_vs_non_fraudulent(df):
    """ Plot count of fraudulent vs non-fraudulent transactions. """
    logger.info("Starting plot: Fraudulent vs Non-Fraudulent Transactions")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Fraudulent vs Non-Fraudulent Transactions (Credit Card Data)')
    plt.show()
    logger.info("Completed plot: Fraudulent vs Non-Fraudulent Transactions")

def plot_purchase_value_distribution(df):
    """ Plot distribution of purchase values for fraud data. """
    logger.info("Starting plot: Purchase Value Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['purchase_value'], bins=50, kde=True)
    plt.title('Distribution of Purchase Values (Fraud Data)')
    plt.show()
    logger.info("Completed plot: Purchase Value Distribution")

def plot_age_distribution(df):
    """ Plot distribution of age for fraud data. """
    logger.info("Starting plot: Age Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title('Distribution of Age (Fraud Data)')
    plt.show()
    logger.info("Completed plot: Age Distribution")

def plot_bivariate_analysis(creditcard_df, fraud_data_df, ip_to_country_df):
    """
    Bivariate Analysis - Creates various plots to analyze the relationships between features.
    """
    # Correlation Heatmap for Credit Card Data
    logger.info("Starting plot: Correlation Heatmap (Credit Card Data)")
    plt.figure(figsize=(12, 8))
    corr_matrix = creditcard_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap (Credit Card Data)')
    plt.show()
    logger.info("Completed plot: Correlation Heatmap (Credit Card Data)")

    # Fraud occurrence by gender in Fraud Data
    logger.info("Starting plot: Fraud Occurrence by Gender")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sex', hue='class', data=fraud_data_df)
    plt.title('Fraud Occurrence by Gender (Fraud Data)')
    plt.show()
    logger.info("Completed plot: Fraud Occurrence by Gender")

    # Time difference between signup and purchase
    logger.info("Starting plot: Time Difference between Signup and Purchase")
    fraud_data_df['time_diff'] = (fraud_data_df['purchase_time'] - fraud_data_df['signup_time']).dt.total_seconds() / 3600
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_data_df['time_diff'], bins=50, kde=True)
    plt.title('Distribution of Time Difference between Signup and Purchase (Fraud Data)')
    plt.show()
    logger.info("Completed plot: Time Difference between Signup and Purchase")

    # Distribution of top countries in IP Address to Country Dataset
    logger.info("Starting plot: Top Countries in IP Address Range")
    plt.figure(figsize=(12, 6))
    top_countries = ip_to_country_df['country'].value_counts().head(10)
    sns.barplot(x=top_countries.index, y=top_countries.values)
    plt.title('Top 10 Countries in IP Address Range')
    plt.xticks(rotation=45)
    plt.show()
    logger.info("Completed plot: Top Countries in IP Address Range")
