import matplotlib.pyplot as plt
import seaborn as sns

def plot_transaction_amount_distribution(df):
    """ Plot distribution of transaction amounts for credit card data. """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Distribution of Transaction Amounts (Credit Card Data)')
    plt.show()

def plot_fraudulent_vs_non_fraudulent(df):
    """ Plot count of fraudulent vs non-fraudulent transactions. """
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Fraudulent vs Non-Fraudulent Transactions (Credit Card Data)')
    plt.show()

def plot_purchase_value_distribution(df):
    """ Plot distribution of purchase values for fraud data. """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['purchase_value'], bins=50, kde=True)
    plt.title('Distribution of Purchase Values (Fraud Data)')
    plt.show()

def plot_age_distribution(df):
    """ Plot distribution of age for fraud data. """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title('Distribution of Age (Fraud Data)')
    plt.show()
