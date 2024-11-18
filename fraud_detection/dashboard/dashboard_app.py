from dash import dcc, html
import dash
import plotly.express as px
from flask import Flask
import requests

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# API URL for fetching statistics (from the Flask API)
API_URL = "http://127.0.0.1:5000/stats"

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'text-align': 'center'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='general-summary'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='age-distribution'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='sex-distribution'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='fraud-trends'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='top-purchases'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='fraud-by-source'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='geography'),
        ], className="card"),

        html.Div([
            dcc.Graph(id='devices-fraud'),
        ], className="card")
    ], className="row")
])

# Callback to update 'general-summary' graph
@app.callback(
    dash.dependencies.Output('general-summary', 'figure'),
    [dash.dependencies.Input('general-summary', 'id')]
)
def update_general_summary_graph(input_value):
    response = requests.get(f'{API_URL}/general_summary')
    data = response.json()

    fig = px.bar(
        x=['Total Users', 'Total Fraud Cases', 'Fraud Rate', 'Average Purchase Value'],
        y=[data['total_users'], data['total_fraud_cases'], data['fraud_rate'], data['average_purchase_value']],
        labels={'x': 'Metrics', 'y': 'Values'},
        title="General Summary"
    )
    return fig

# Callback to update 'age-distribution' graph
@app.callback(
    dash.dependencies.Output('age-distribution', 'figure'),
    [dash.dependencies.Input('age-distribution', 'id')]
)
def update_age_distribution_graph(input_value):
    response = requests.get(f'{API_URL}/age_distribution')
    data = response.json()

    fig = px.bar(
        data,
        x='age',
        y='class',
        labels={'class': 'Fraud Cases', 'age': 'Age'},
        title="Fraud Cases by Age"
    )
    return fig

# Callback to update 'sex-distribution' graph
@app.callback(
    dash.dependencies.Output('sex-distribution', 'figure'),
    [dash.dependencies.Input('sex-distribution', 'id')]
)
def update_sex_distribution_graph(input_value):
    response = requests.get(f'{API_URL}/sex_distribution')
    data = response.json()

    fig = px.bar(
        data,
        x='sex',
        y='class',
        labels={'class': 'Fraud Cases', 'sex': 'Sex'},
        title="Fraud Cases by Sex"
    )
    return fig

# Callback to update 'fraud-trends' graph
@app.callback(
    dash.dependencies.Output('fraud-trends', 'figure'),
    [dash.dependencies.Input('fraud-trends', 'id')]
)
def update_fraud_trends_graph(input_value):
    response = requests.get(f'{API_URL}/trends')
    data = response.json()

    fig = px.line(
        data,
        x='purchase_date',
        y='class',
        title="Fraud Trends Over Time",
        labels={'purchase_date': 'Date', 'class': 'Fraud Cases'}
    )
    return fig

# Callback to update 'top-purchases' graph
@app.callback(
    dash.dependencies.Output('top-purchases', 'figure'),
    [dash.dependencies.Input('top-purchases', 'id')]
)
def update_top_purchases_graph(input_value):
    response = requests.get(f'{API_URL}/top_purchases')
    data = response.json()

    fig = px.bar(
        data,
        x='user_id',
        y='purchase_value',
        labels={'user_id': 'User ID', 'purchase_value': 'Purchase Value'},
        title="Top 10 Purchases by Users"
    )
    return fig

# Callback to update 'fraud-by-source' graph
@app.callback(
    dash.dependencies.Output('fraud-by-source', 'figure'),
    [dash.dependencies.Input('fraud-by-source', 'id')]
)
def update_fraud_by_source_graph(input_value):
    response = requests.get(f'{API_URL}/fraud_by_source')
    data = response.json()

    fig = px.bar(
        data,
        x='source',
        y='class',
        labels={'source': 'Source', 'class': 'Fraud Rate'},
        title="Fraud Rate by Source"
    )
    return fig

# Callback to update 'geography' graph
@app.callback(
    dash.dependencies.Output('geography', 'figure'),
    [dash.dependencies.Input('geography', 'id')]
)
def update_geography_graph(input_value):
    response = requests.get(f'{API_URL}/geography')
    data = response.json()

    fig = px.scatter(
        data,
        x='ip_address',
        y='class',
        labels={'ip_address': 'IP Address', 'class': 'Fraud Cases'},
        title="Geographic Distribution of Fraud Cases"
    )
    return fig

# Callback to update 'devices-fraud' graph
@app.callback(
    dash.dependencies.Output('devices-fraud', 'figure'),
    [dash.dependencies.Input('devices-fraud', 'id')]
)
def update_devices_fraud_graph(input_value):
    response = requests.get(f'{API_URL}/devices_fraud')
    data = response.json()

    fig = px.bar(
        data,
        x='device_id',
        y='class',
        color='browser',
        labels={'device_id': 'Device ID', 'class': 'Fraud Cases', 'browser': 'Browser'},
        title="Fraud Cases by Device and Browser"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
