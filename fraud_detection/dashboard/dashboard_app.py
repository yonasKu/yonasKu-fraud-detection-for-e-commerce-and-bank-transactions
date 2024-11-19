from dash import dcc, html
import dash
import plotly.express as px
from flask import Flask
import requests
import pandas as pd
# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# API URL for fetching statistics (from the Flask API)
API_URL = "http://127.0.0.1:5000/stats"

# Layout of the dashboard
# Define the app layout with tabs
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'text-align': 'center'}),

    # Tabs for switching between Statistics and Prediction views
    dcc.Tabs([
        dcc.Tab(label='Statistics', children=[
            html.Div([
                html.H2("Overview"),
                html.P("Explore statistics related to fraud cases, user demographics, fraud sources, and trends."),
                
                # Graphs in card-style divs with descriptions
                html.Div([
                    html.Div([
                        html.H4("General Summary"),
                        html.P("This chart shows a general summary of users, fraud cases, and fraud rate."),
                        dcc.Graph(id='general-summary')
                    ], className="card"),

                    html.Div([
                        html.H4("Age Distribution"),
                        html.P("View the distribution of fraud cases across different age groups."),
                        dcc.Graph(id='age-distribution')
                    ], className="card"),

                    html.Div([
                        html.H4("Sex Distribution"),
                        html.P("Shows the proportion of fraud cases across different genders."),
                        dcc.Graph(id='sex-distribution')
                    ], className="card"),

                    html.Div([
                        html.H4("Fraud Trends"),
                        html.P("Observe how fraud cases have trended over time."),
                        dcc.Graph(id='fraud-trends')
                    ], className="card"),

                    html.Div([
                        html.H4("Fraud by Source"),
                        html.P("Distribution of fraud cases based on source platforms."),
                        dcc.Graph(id='fraud-by-source')
                    ], className="card"),

                    html.Div([
                        html.H4("Geography of Fraud"),
                        html.P("Shows the geographical distribution of fraud cases."),
                        dcc.Graph(id='geography')
                    ], className="card"),

                    html.Div([
                        html.H4("Devices Used in Fraud"),
                        html.P("Top devices involved in fraud cases and their browsers."),
                        dcc.Graph(id='devices-fraud')
                    ], className="card")
                ], className="row")
            ])
        ]),

        dcc.Tab(label='Prediction', children=[
            html.Div([
                html.H2("Fraud Prediction", style={'text-align': 'center', 'margin-bottom': '20px'}),
                html.P("Input data to make predictions on potential fraud cases.",
                       style={'text-align': 'center', 'font-size': '16px', 'color': '#555'}),
                
                html.Div([
                    html.Div([
                        html.Label("User ID", style={'font-weight': 'bold'}),
                        dcc.Input(id='user-id', type='number', placeholder="Enter user ID",
                                  style={'width': '100%', 'padding': '8px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Signup Time", style={'font-weight': 'bold'}),
                        dcc.DatePickerSingle(id='signup-time', placeholder="Select signup date",
                                             style={'width': '100%'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Purchase Time", style={'font-weight': 'bold'}),
                        dcc.DatePickerSingle(id='purchase-time', placeholder="Select purchase date",
                                             style={'width': '100%'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Purchase Value", style={'font-weight': 'bold'}),
                        dcc.Input(id='purchase-value', type='number', placeholder="Enter purchase value",
                                  style={'width': '100%', 'padding': '8px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Device ID", style={'font-weight': 'bold'}),
                        dcc.Input(id='device-id', type='text', placeholder="Enter device ID",
                                  style={'width': '100%', 'padding': '8px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Source", style={'font-weight': 'bold'}),
                        dcc.Dropdown(id='source', options=[
                            {'label': 'SEO', 'value': 'SEO'},
                            {'label': 'Ads', 'value': 'Ads'},
                            {'label': 'Direct', 'value': 'Direct'}
                        ], placeholder="Select source",
                        style={'border-radius': '5px'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Browser", style={'font-weight': 'bold'}),
                        dcc.Dropdown(id='browser', options=[
                            {'label': 'Chrome', 'value': 'Chrome'},
                            {'label': 'Firefox', 'value': 'Firefox'},
                            {'label': 'Opera', 'value': 'Opera'},
                            {'label': 'Safari', 'value': 'Safari'}
                        ], placeholder="Select browser",
                        style={'border-radius': '5px'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Sex", style={'font-weight': 'bold'}),
                        dcc.Dropdown(id='sex', options=[
                            {'label': 'Male', 'value': 'M'},
                            {'label': 'Female', 'value': 'F'}
                        ], placeholder="Select gender",
                        style={'border-radius': '5px'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("Age", style={'font-weight': 'bold'}),
                        dcc.Input(id='age', type='number', placeholder="Enter age",
                                  style={'width': '100%', 'padding': '8px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Div([
                        html.Label("IP Address", style={'font-weight': 'bold'}),
                        dcc.Input(id='ip-address', type='text', placeholder="Enter IP address",
                                  style={'width': '100%', 'padding': '8px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                    ], style={'margin-bottom': '15px'}),

                    html.Button("Submit", id='submit-button', n_clicks=0,
                                style={
                                    'background-color': '#007BFF', 
                                    'color': 'white', 
                                    'padding': '10px 15px', 
                                    'border': 'none', 
                                    'border-radius': '5px', 
                                    'cursor': 'pointer',
                                    'font-size': '16px'
                                })
                ], style={
                    'max-width': '500px', 
                    'margin': '0 auto', 
                    'padding': '20px', 
                    'border': '1px solid #ddd', 
                    'border-radius': '10px',
                    'background-color': '#f9f9f9',
                    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                }),

                html.Div(id='output-display', style={
                    'margin-top': '30px', 
                    'text-align': 'center', 
                    'font-size': '16px',
                    'color': '#333'
                })
            ], style={'padding': '20px'})
        ])
    ])
])

# Callback to display user input
@app.callback(
    dash.dependencies.Output('output-display', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [
        dash.dependencies.State('user-id', 'value'),
        dash.dependencies.State('signup-time', 'date'),
        dash.dependencies.State('purchase-time', 'date'),
        dash.dependencies.State('purchase-value', 'value'),
        dash.dependencies.State('device-id', 'value'),
        dash.dependencies.State('source', 'value'),
        dash.dependencies.State('browser', 'value'),
        dash.dependencies.State('sex', 'value'),
        dash.dependencies.State('age', 'value'),
        dash.dependencies.State('ip-address', 'value')
    ]
)
def handle_submit(n_clicks, user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address):
    if n_clicks > 0:
        return html.Div([
            html.H4("Submitted Data:"),
            html.P(f"User ID: {user_id}"),
            html.P(f"Signup Time: {signup_time}"),
            html.P(f"Purchase Time: {purchase_time}"),
            html.P(f"Purchase Value: {purchase_value}"),
            html.P(f"Device ID: {device_id}"),
            html.P(f"Source: {source}"),
            html.P(f"Browser: {browser}"),
            html.P(f"Sex: {sex}"),
            html.P(f"Age: {age}"),
            html.P(f"IP Address: {ip_address}")
        ])
    return "Please fill out the form and submit."
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

    fig = px.pie(
        data,
        names='sex',
        values='class',
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
# Callback to update 'devices-fraud' graph
@app.callback(
    dash.dependencies.Output('devices-fraud', 'figure'),
    [dash.dependencies.Input('devices-fraud', 'id')]
)

def update_devices_fraud_graph(input_value):
    response = requests.get(f'{API_URL}/devices_fraud')
    data = response.json()

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(data)

    # Aggregate the data by 'device_id', summing 'class' and taking the first browser
    aggregated_df = df.groupby('device_id').agg(
        total_fraud=('class', 'sum'),
        browser=('browser', 'first')  # Take the first browser for each device
    ).reset_index()

    # Get the top 10 devices by fraud cases
    top_devices = aggregated_df.nlargest(10, 'total_fraud')

    # Plot the bar chart
    fig = px.bar(
        top_devices,
        y='device_id',
        x='total_fraud',
        color='browser',
        labels={'device_id': 'Device ID', 'total_fraud': 'Fraud Cases', 'browser': 'Browser'},
        title="Top 10 Fraud Cases by Device and Browser",
        orientation='h'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
