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
from prophet.plot import plot_plotly, plot_components_plotly

# Setup mlflow tracking
url = '127.0.0.1'
port = '8080'
mlflow.set_tracking_uri(uri=f"http://{url}:{port}")

# Display MLFlow link
st.title('Forecast Model Training')
st.markdown(f'[MLFlow url]({mlflow.get_tracking_uri()})')

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

# Update stock data
st.subheader('Update Data')
st.text('Update stock data with latest information.')
if st.checkbox("Update Stock Data"):
    # Get polygon key
    polygon_api_key = json.load(open('keys.json'))['POLYGON_API_KEY']

    # Get min date
    min_date = utils.get_max_date(file_dir='data/daily-aggregates')

    # First time min date
    # min_date = (datetime.datetime.now() - datetime.timedelta(days=5 * 365)).strftime('%Y-%m-%d')

    # Download stock data
    utils.update_stock_data(
        min_date=min_date.strftime('%Y-%m-%d'),
        max_date=datetime.datetime.now().strftime('%Y-%m-%d'),
        api_key=polygon_api_key
    )

    stock_df = utils.create_stock_dataframe(file_dir='data/daily-aggregates')

    # Sort data by date
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.sort_values(by=['Date'], inplace=True, ascending=False)

    # Save dataframe
    stock_df.to_csv('data/stock_data/data.csv', index=False)
st.divider()

# Train model
st.subheader('Model Training')
st.text('Train a model to forecast stock prices, WARGNING: This may take a while.')
if st.checkbox("Train Model"):
    # Load stock data if not already loaded
    try:
        stock_df
    except Exception as e:
        print(e)
        stock_df = pd.read_csv('data/stock_data/data.csv')
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Limit stocks to those with at least 1000 daily summaries
    forecast_symbols = stock_df.Exchange_Symbol.value_counts()[(stock_df.Exchange_Symbol.value_counts() > 1000).values]
    forecast_symbols = list(forecast_symbols.keys())

    # Place holder for parameters
    param_list = []
    for symbol in forecast_symbols:
        param_list.append({
            'stock_symbol': symbol,
            'df': stock_df
        })

    # params = []
    pool = mp.Pool(mp.cpu_count()-1)
    params = pool.map(utils.create_forecast_params, param_list)
    pool.close()

    # Create nested runs
    experiment_id = mlflow.create_experiment(f"experiment-{datetime.datetime.now().strftime('%Y%m%d')}")
    for param in params:
        param['experiment_id'] = experiment_id
        param['tracking_uri'] = mlflow.get_tracking_uri()

    with mlflow.start_run(
        run_name=f"stock-predictions-{datetime.datetime.now().strftime('%Y%m%d')}",
        experiment_id=experiment_id,
        description="parent"
    ) as parent_run:
        mlflow.log_param("parent", "yes")
        pool = mp.Pool(processes=mp.cpu_count()-1)
        pool.map(utils.train_model, params)
    pool.close()
st.divider()