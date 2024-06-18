import os
import sys
import json
import utils
import mlflow
import prophet
import datetime
import requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import multiprocessing as mp

from prophet import Prophet
from utils import nav_page
from prophet.plot import plot_plotly, plot_components_plotly

# Setup mlflow tracking
url = '127.0.0.1'
port = '8080'
mlflow.set_tracking_uri(uri=f"http://{url}:{port}")

# Get Polygon API data
file_path = os.path.join(os.getcwd().replace('/pages', ''), 'keys.json')
with open(file_path, 'r') as file:
    keys_dict = json.load(file)
polygon_api_key = keys_dict['POLYGON_API_KEY']

@st.cache_data
def load_stock_data(file_path:str):
    stock_df = pd.read_csv(file_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    return stock_df

st.set_page_config(
    page_title="Tate Investment Managment",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Introduction
st.header("Investment Management Dashboard")
st.markdown(
    """
    Welcome to the Tate Investment Management dashboard!
    Select a model series to get started.
    """
)
# Evaluate model
st.subheader('Stock Forecasts')

# Select experiment and load data
experiments = [e.experiment_id for e in mlflow.search_experiments() if e.name != "Default"]
experiment_id = st.selectbox('Select model series', experiments)

# Load model data
model_runs = mlflow.search_runs(experiment_ids=experiment_id, filter_string = 'attributes.status = "FINISHED"')
model_runs['stock_symbol'] = model_runs['tags.mlflow.runName'].apply(lambda x: x.split('-')[0])

# Display overall training metrics
st.markdown('Forecast Accuracy')

# Display individual model metrics
display_df = model_runs[['stock_symbol', 'metrics.rmse', 'metrics.mape', 'metrics.mse']]
display_df = display_df.rename(columns = {'stock_symbol': 'Stock Symbol', 'metrics.rmse': 'RMSE', 'metrics.mape': 'MAPE', 'metrics.mse': 'MSE'})
st.dataframe(display_df, hide_index=True)

# Load stock symbols
stock_symbols = list(model_runs['tags.mlflow.runName'])
stock_symbols = [x.split('-')[0] for x in stock_symbols]

# Load data
stock_df = load_stock_data('data/stock_data/data.csv')

# Simulate selecting stock symbol
stock_symbol = st.selectbox('Select Stock Symbol', stock_symbols)
if 'stock_symbol' not in st.session_state:
    st.session_state['stock_symbol'] = stock_symbol

stock_details = utils._get_ticker_details(
    date=datetime.datetime.now().strftime('%Y-%m-%d'),
    stock_symbol=stock_symbol,
    api_key=polygon_api_key
)
try:
    listing_date = stock_details['results']['list_date']
except:
    listing_date = 'N/A'

if stock_details['results']['type'] == 'CS':
    st.markdown(
        f"""
        ##### {stock_details['results']['name']}
        - **Currency**: {stock_details['results']['currency_name']}
        - **Locale**: {stock_details['results']['locale']}
        - **Market Cap**: {stock_details['results']['market_cap']:,.0f}
        - **Primary Exchange**: {stock_details['results']['primary_exchange']}
        - **Company Type**: {stock_details['results']['sic_description']}
        - **Listing Date**: {listing_date}
        - {stock_details['results']['homepage_url']}

        {stock_details['results']['description']}
        """
    )
elif stock_details['results']['type'] == 'ETF':
    st.markdown(
        f"""
        ##### {stock_details['results']['name']}
        - **Currency**: {stock_details['results']['currency_name']}
        - **Locale**: {stock_details['results']['locale']}
        - **Primary Exchange**: {stock_details['results']['primary_exchange']}
        - **Share Class Shares Outstanding**: {stock_details['results']['share_class_shares_outstanding']:,.0f}
        - **Company Type**: {stock_details['results']['type']}
        - **Listing Date**: {listing_date}
        """
    )
elif stock_details['results']['type'] == 'FUND':
    st.markdown(
        f"""
        ##### {stock_details['results']['name']}
        - **Currency**: {stock_details['results']['currency_name']}
        - **Locale**: {stock_details['results']['locale']}
        - **Market Cap**: {stock_details['results']['market_cap']:,.0f}
        - **Primary Exchange**: {stock_details['results']['primary_exchange']}
        - **Company Type**: {stock_details['results']['type']}
        - **Listing Date**: {listing_date}
        - {stock_details['results']['homepage_url']}

        {stock_details['results']['description']}
        """
    )
elif stock_details['results']['type'] == 'UNIT':
    st.markdown(
        f"""
        ##### {stock_details['results']['name']}
        - **Currency**: {stock_details['results']['currency_name']}
        - **Locale**: {stock_details['results']['locale']}
        - **Primary Exchange**: {stock_details['results']['primary_exchange']}
        - **Company Type**: {stock_details['results']['sic_description']}
        """
    )

# filter table based on selected stock
model_path = model_runs[model_runs['tags.mlflow.runName'].str.contains(stock_symbol)]['artifact_uri'].values[0]
model_path = os.path.join(model_path, 'prophet-model')
model = mlflow.prophet.load_model(model_path)

# Display performance metrics
forecast_period = st.number_input('Forecast Period', min_value=1, max_value=365, value=30)

forecast = model.predict(model.make_future_dataframe(periods=forecast_period))
st.plotly_chart(plot_plotly(model, forecast, xlabel='Date', ylabel='Close Price'))