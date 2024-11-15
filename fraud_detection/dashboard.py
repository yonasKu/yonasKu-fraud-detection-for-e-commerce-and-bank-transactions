import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify
from dash import Dash, dcc, html
import dash.dependencies as dd
import plotly.express as px

# Initialize Flask app
server = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load fraud data
data_path = '../data/Fraud_Data.csv'  # Adjust the path as necessary
df = pd.read_csv(data_path)

# Ensure 'signup_time' and 'purchase_time' are in datetime format
df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

# Calculate transaction frequency and transaction velocity for the entire dataset
df['transaction_frequency'] = (df['purchase_time'] - df['signup_time']).dt.days / 30  # Monthly frequency
df['transaction_velocity'] = df['purchase_value'] / df['transaction_frequency'].replace(0, 1)  # Avoid division by zero

# Prepare data for dashboard
def get_summary_statistics():
    total_transactions = len(df)
    total_fraud_cases = df['class'].sum()  # Assuming 'class' column indicates fraud (1 = fraud, 0 = not fraud)
    fraud_percentage = (total_fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0
    return {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }

# Load credit fraud detection model
credit_fraud_model_path = 'model/Credit Card Fraud Detection with Random Forest_random_forest_model.pkl'
try:
    credit_fraud_model = joblib.load(credit_fraud_model_path)
except FileNotFoundError:
    credit_fraud_model = None
    logger.error(f"Credit fraud model file not found at path: {credit_fraud_model_path}")

# Load general fraud detection model
general_fraud_model_path = 'model/Fraud Detection with Random Forest_random_forest_model.pkl'
try:
    general_fraud_model = joblib.load(general_fraud_model_path)
except FileNotFoundError:
    general_fraud_model = None
    logger.error(f"General fraud model file not found at path: {general_fraud_model_path}")

# Data preprocessing function to match model input
def preprocess_input(data):
    # Convert device_id and ip_address to numeric (or other encoding as needed)
    device_id = hash(data['device_id']) % 100000  # Example: hash and reduce to unique numeric id
    browser = {"Chrome": 1, "Safari": 2, "Firefox": 3}.get(data['browser'], 0)  # Simple encoding example
    source = {"SEO": 1, "Ads": 2, "Direct": 3}.get(data['source'], 0)
    sex = 1 if data['sex'] == 'M' else 0

    # Handle age with normalization and default value if missing
    if 'age' in data:
        age = (data['age'] - df['age'].mean()) / df['age'].std()  # Normalized age
    else:
        age = 0  # Default normalized value for age if missing

    # Convert datetime features
    signup_time = pd.to_datetime(data['signup_time'])
    purchase_time = pd.to_datetime(data['purchase_time'])
    transaction_frequency = (purchase_time - signup_time).days / 30  # Example calculation
    transaction_velocity = data['purchase_value'] / transaction_frequency if transaction_frequency > 0 else 0
    hour_of_day = purchase_time.hour
    day_of_week = purchase_time.dayofweek

    # Prepare the processed input
    processed_data = {
        "user_id": data['user_id'],
        "purchase_value": data['purchase_value'],
        "device_id": device_id,
        "source": source,
        "browser": browser,
        "sex": sex,
        "age": age,
        "country": int(data.get('country', 0)),  # Assuming some encoding for country
        "transaction_frequency": transaction_frequency,
        "transaction_velocity": transaction_velocity,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week
    }
    return processed_data

# Define routes for model predictions
@server.route('/predict/credit_fraud', methods=['POST'])
def predict_credit_fraud():
    data = request.get_json()
    processed_data = preprocess_input(data)
    input_df = pd.DataFrame([processed_data])
    if credit_fraud_model:
        predictions = credit_fraud_model.predict(input_df)
        return jsonify(predictions.tolist()), 200
    else:
        return jsonify({'error': 'Credit fraud model not loaded'}), 500

@server.route('/predict/general_fraud', methods=['POST'])
def predict_general_fraud():
    data = request.get_json()
    processed_data = preprocess_input(data)
    input_df = pd.DataFrame([processed_data])
    if general_fraud_model:
        predictions = general_fraud_model.predict(input_df)
        return jsonify(predictions.tolist()), 200
    else:
        return jsonify({'error': 'General fraud model not loaded'}), 500

# Initialize Dash app
app = Dash(__name__, server=server, suppress_callback_exceptions=True)

# Layout for the dashboard
app.layout = html.Div([
    html.H1("Fraud Insights Dashboard"),
    html.Div(id='summary-boxes', style={'display': 'flex', 'justify-content': 'space-around'}),
    dcc.Graph(id='fraud-trend-chart'),
    dcc.Graph(id='fraud-by-sex-chart'),
    dcc.Graph(id='fraud-by-age-chart'),
    dcc.Graph(id='device-fraud-chart'),
    dcc.Graph(id='browser-fraud-chart'),
    dcc.Graph(id='hourly-fraud-chart'),  # New: Fraud by hour
    dcc.Graph(id='daily-fraud-chart'),   # New: Fraud by day
    dcc.Graph(id='frequency-distribution-chart'),  # New: Transaction Frequency
    dcc.Graph(id='velocity-distribution-chart'),   # New: Transaction Velocity
   ])

# Callback to update summary boxes
@app.callback(
    dd.Output('summary-boxes', 'children'),
    [dd.Input('fraud-trend-chart', 'id')]
)
def update_summary_boxes(_):
    summary_stats = get_summary_statistics()
    return [
        html.Div([html.H3("Total Transactions"), html.P(str(summary_stats['total_transactions']))]),
        html.Div([html.H3("Total Fraud Cases"), html.P(str(summary_stats['total_fraud_cases']))]),
        html.Div([html.H3("Fraud Percentage"), html.P(f"{summary_stats['fraud_percentage']:.2f}%")]),
    ]

# Callback to update fraud trend chart
@app.callback(
    dd.Output('fraud-trend-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_fraud_trend_chart(_):
    trends = df.groupby(df['signup_time'].dt.to_period('M'))['class'].sum().reset_index()
    trends['signup_time'] = trends['signup_time'].astype(str)
    trends.columns = ['Month', 'Fraud Cases']
    fig = px.line(trends, x='Month', y='Fraud Cases', title='Fraud Cases Over Time')
    return fig

# Callback to update fraud by sex chart
@app.callback(
    dd.Output('fraud-by-sex-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_fraud_by_sex_chart(_):
    sex_counts = df[df['class'] == 1]['sex'].value_counts().reset_index()
    sex_counts.columns = ['Sex', 'Fraud Cases']
    fig = px.pie(sex_counts, names='Sex', values='Fraud Cases', title='Fraud Cases by Sex')
    return fig

# Callback to update fraud by age chart
@app.callback(
    dd.Output('fraud-by-age-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_fraud_by_age_chart(_):
    fig = px.histogram(df[df['class'] == 1], x='age', nbins=20, title='Fraud Cases by Age')
    fig.update_layout(xaxis_title='Age', yaxis_title='Count')
    return fig

# Callback to update transaction frequency distribution chart
@app.callback(
    dd.Output('frequency-distribution-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_frequency_distribution_chart(_):
    fig = px.histogram(df, x='transaction_frequency', nbins=20, title='Transaction Frequency Distribution')
    fig.update_layout(xaxis_title='Transaction Frequency (months)', yaxis_title='Count')
    return fig

# Callback to update transaction velocity distribution chart
@app.callback(
    dd.Output('velocity-distribution-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_velocity_distribution_chart(_):
    fig = px.histogram(df, x='transaction_velocity', nbins=20, title='Transaction Velocity Distribution')
    fig.update_layout(xaxis_title='Transaction Velocity', yaxis_title='Count')
    return fig

# Callback to update device-fraud chart
@app.callback(
    dd.Output('device-fraud-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_device_fraud_chart(_):
    device_counts = df[df['class'] == 1]['device_id'].value_counts().reset_index()
    device_counts.columns = ['Device', 'Fraud Cases']
    fig = px.bar(device_counts, x='Device', y='Fraud Cases', title='Fraud Cases by Device')
    return fig

# Callback to update browser-fraud chart
@app.callback(
    dd.Output('browser-fraud-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_browser_fraud_chart(_):
    browser_counts = df[df['class'] == 1]['browser'].value_counts().reset_index()
    browser_counts.columns = ['Browser', 'Fraud Cases']
    fig = px.bar(browser_counts, x='Browser', y='Fraud Cases', title='Fraud Cases by Browser')
    return fig

# Callback to update hourly fraud chart
@app.callback(
    dd.Output('hourly-fraud-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_hourly_fraud_chart(_):
    hourly_counts = df[df['class'] == 1]['purchase_time'].dt.hour.value_counts().sort_index()
    fig = px.bar(hourly_counts, x=hourly_counts.index, y=hourly_counts.values, title='Fraud Cases by Hour of Day')
    fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Fraud Cases')
    return fig

# Callback to update daily fraud chart
@app.callback(
    dd.Output('daily-fraud-chart', 'figure'),
    [dd.Input('summary-boxes', 'children')]
)
def update_daily_fraud_chart(_):
    daily_counts = df[df['class'] == 1]['purchase_time'].dt.day.value_counts().sort_index()
    fig = px.bar(daily_counts, x=daily_counts.index, y=daily_counts.values, title='Fraud Cases by Day of Month')
    fig.update_layout(xaxis_title='Day of Month', yaxis_title='Fraud Cases')
    return fig


if __name__ == '__main__':
    server.run(debug=True)
