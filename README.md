# Fraud Detection System

## Overview
Fraud Detection is an essential task across industries like finance, e-commerce, and insurance. This project implements a scalable and modular fraud detection system using advanced machine learning techniques. The system supports real-time fraud prediction via APIs, provides a visual dashboard for fraud analysis, and ensures model explainability with interpretability tools.

## Key Features
- **Data Preprocessing and Feature Engineering**: Automated pipelines for data cleaning, transformation, and feature extraction.
- **Machine Learning Models**: Pre-trained models (Random Forest, XGBoost) optimized for fraud detection.
- **Explainability**: SHAP and LIME integration for feature importance and decision explainability.
- **Interactive Dashboard**: Visualize fraud trends and distribution using Dash.
- **Real-Time API**: Predict fraud in real time with a Flask API.
- **CI/CD and MLOps**: Automated testing, deployment pipelines, and model monitoring.

## Project Structure
fraud_detection/
├── data/ # Raw and processed datasets
├── src/ # Core codebase
│ ├── preprocessing/ # Data cleaning and feature engineering
│ ├── training/ # Model training and evaluation
│ ├── api/ # Flask API for fraud prediction
│ ├── dashboard/ # Dash-based visualization
│ └── utils/ # Shared utilities
├── models/ # Saved trained models
├── tests/ # Unit and integration tests
├── notebooks/ # Jupyter notebooks for EDA and prototyping
├── logs/ # Logs for debugging
├── requirements.txt # Dependency file
├── Dockerfile # Docker configuration
└── README.md # Project documentation



## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
2. Install Dependencies
Ensure Python 3.8+ is installed, then install dependencies:


pip install -r requirements.txt
3. Run the Pipeline
a. Preprocess the Data


python src/preprocessing/data_cleaning.py
python src/preprocessing/feature_engineering.py
b. Train and Evaluate Models


python src/training/model_training.py
python src/training/model_evaluation.py
4. Start the Flask API
Launch the API for real-time fraud detection:


python src/api/flask_app.py
The API will run on: http://127.0.0.1:5000

5. Start the Dashboard
Run the interactive dashboard:


python src/dashboard/dashboard_app.py
Access the dashboard at: http://127.0.0.1:8050

6. Run Tests
Validate functionality using unit and integration tests:


pytest tests/
Using the Flask API
Endpoints
1. Predict Fraud
URL: /predict
Method: POST
Description: Predict whether a transaction is fraudulent.
Sample Request:


{
    "transaction_id": "12345",
    "amount": 500,
    "ip_address": "192.168.0.1",
    "user_agent": "Mozilla/5.0",
    "device_type": "Mobile",
    "location": "USA"
}
## Data Setup

The required datasets for this project are not included in the repository due to privacy and size constraints. Please download the necessary files from the provided Google Drive link and place them in the `data/` directory as follows:

### Step-by-Step Instructions

#### 1. Download the Data Files:
Download the datasets from the following Google Drive link:
[Google Drive - Fraud Detection Data](link-to-your-google-drive)

#### 2. Create the data/ Directory:
Inside the cloned repository, create a folder named data:
```bash
mkdir data
3. Move the Files:
After downloading the files, place them in the data/ directory:


fraud_detection/
├── data/
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
Dataset Descriptions
Fraud_Data.csv
Primary transaction dataset containing historical fraud data
Features include transaction amount, timestamp, merchant details, etc.
Target variable: is_fraud (0: legitimate, 1: fraudulent)

IpAddress_to_Country.csv
IP address to country mapping for geolocation features
Used for enriching transaction data with location information

creditcard.csv
Additional credit card transaction dataset
Contains anonymized credit card transactions

