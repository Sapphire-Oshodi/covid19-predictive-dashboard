import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Load data only once
data = pd.read_csv("data/country_wise_latest.csv")
data1 = pd.read_csv("data/day_wise.csv")

# Load LSTM model
lstm_model = load_model("models/lstm_model.h5")

# Initialize the app
app = Dash(__name__)
server = app.server  # For deployment with gunicorn

# Data preparation
data['Country/Region'] = data['Country/Region'].astype(str)
data1['Date'] = pd.to_datetime(data1['Date'], format='%Y-%m-%d')
data1.set_index('Date', inplace=True)

# ARIMA model setup
confirmed_cases = data1['Confirmed']
train_size = int(len(confirmed_cases) * 0.8)
train, test = confirmed_cases[:train_size], confirmed_cases[train_size:]

# Refitting ARIMA model on the training data
arima_model_fit = ARIMA(train, order=(5,1,0))  # You can adjust (5,1,0) to your preferred ARIMA parameters
arima_model_fit = arima_model_fit.fit()

# Forecast using ARIMA model
arima_forecast = arima_model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
future_forecast_arima = arima_model_fit.forecast(steps=30)

# LSTM model setup
scaler = MinMaxScaler()
scaled_cases = scaler.fit_transform(confirmed_cases.values.reshape(-1, 1))

sequence_length = 14
X, y = [], []
for i in range(len(scaled_cases) - sequence_length):
    X.append(scaled_cases[i:i + sequence_length])
    y.append(scaled_cases[i + sequence_length])

X, y = np.array(X), np.array(y)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lstm_predictions = lstm_model.predict(X_test)
lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)

# Layout with "Select Country" functionality and bar chart
app.layout = html.Div([
    html.H1("COVID-19 Dashboard", style={'textAlign': 'center', 'color': '#004d99'}),
    html.H3("Analysis by Sapphire Oshodi (DataChicGirl)", style={
        'textAlign': 'center',
        'color': '#666',
        'fontFamily': 'Arial, sans-serif',
        'fontStyle': 'italic',
        'marginTop': '10px'
    }),
    dcc.Tabs([
        dcc.Tab(label="Global Overview", children=[
            dcc.Graph(
                id='covid-map',
                figure=px.choropleth(
                    data,
                    locations="Country/Region",
                    locationmode="country names",
                    color="Confirmed",
                    hover_name="Country/Region",
                    title="Global COVID-19 Cases",
                    color_continuous_scale="Reds"
                )
            ),
            html.Div([
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in data['Country/Region'].unique()],
                    value=None,
                    placeholder="Select a Country",
                    style={'width': '50%', 'margin': 'auto'}
                ),
                html.Button("Select Country", id="select-country-btn", n_clicks=0, style={
                    'margin': '20px auto', 'display': 'block', 'textAlign': 'center', 'width': '20%'
                })
            ]),
            dcc.Graph(id='country-bar-chart', style={'marginTop': '20px'})
        ]),
        dcc.Tab(label="ARIMA Model Analysis", children=[
            dcc.Graph(id='arima-forecast')
        ]),
        dcc.Tab(label="LSTM Model Analysis", children=[
            dcc.Graph(id='lstm-forecast')
        ])
    ]),
    html.Div([
        html.P("Crafted with passion and precision by DataChicGirl.", style={
            'textAlign': 'center',
            'color': '#333',
            'fontFamily': 'Arial, sans-serif',
            'marginTop': '20px'
        })
    ])
])

# Callback to update bar chart and ARIMA/LSTM graphs based on selected country
@app.callback(
    [Output('country-bar-chart', 'figure'),
     Output('arima-forecast', 'figure'),
     Output('lstm-forecast', 'figure')],
    [Input('select-country-btn', 'n_clicks')],
    [State('country-dropdown', 'value')]
)
def update_graphs(n_clicks, selected_country):
    if selected_country:
        filtered_data = data[data['Country/Region'] == selected_country]
        country_stats = filtered_data[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
        country_stats.columns = ['Category', 'Value']
    else:
        filtered_data = data
        country_stats = data[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
        country_stats.columns = ['Category', 'Value']

    # Update bar chart
    bar_chart = px.bar(
        country_stats,
        x='Category',
        y='Value',
        color='Category',
        title=f"COVID-19 Statistics for {selected_country}" if selected_country else "Global COVID-19 Statistics",
        labels={'Value': 'Count', 'Category': 'Statistic'},
        color_discrete_map={'Confirmed': 'blue', 'Deaths': 'red', 'Recovered': 'green'}
    )

    # ARIMA forecast plot
    arima_fig = go.Figure([
        go.Scatter(x=test.index, y=test, mode='lines', name='Actual'),
        go.Scatter(x=test.index, y=arima_forecast, mode='lines', name='ARIMA Forecast', line=dict(color='red'))
    ])

    # LSTM forecast plot
    lstm_fig = go.Figure([
        go.Scatter(x=test.index, y=test.values, mode='lines', name='Actual'),
        go.Scatter(x=test.index, y=lstm_predictions_rescaled.flatten(), mode='lines', name='LSTM Predictions', line=dict(color='blue'))
    ])

    return bar_chart, arima_fig, lstm_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
