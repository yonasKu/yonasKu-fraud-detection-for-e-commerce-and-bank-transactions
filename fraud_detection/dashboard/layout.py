from dash import html, dcc

def create_dashboard_layout():
    return html.Div([
        html.H1("Fraud Detection Dashboard", style={'text-align': 'center', 'padding': '10px'}),
        
        # Summary Metrics Section
        html.Div([
            html.Div([dcc.Graph(id='general-summary')], className="card"),
            html.Div([dcc.Graph(id='fraud-trends')], className="card"),
        ], className="row"),

        # Distribution Analysis Section
        html.Div([
            html.Div([dcc.Graph(id='age-distribution')], className="card"),
            html.Div([dcc.Graph(id='sex-distribution')], className="card"),
        ], className="row"),

        # Top Purchases Section
        html.Div([
            html.Div([dcc.Graph(id='top-purchases')], className="card"),
            html.Div([dcc.Graph(id='fraud-by-source')], className="card"),
        ], className="row"),

        # Additional Analysis
        html.Div([
            html.Div([dcc.Graph(id='device-fraud')], className="card"),
            html.Div([dcc.Graph(id='geographic-fraud')], className="card"),
        ], className="row"),
    ], className="container")
