import os
import sys
import json
import utils
import datetime
import requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title('Stocks')
st.markdown(
    """
    This page contains stock data such as historic prices, trading volume, and other news.
    """
)

# Get date
min_date = datetime.datetime.now() - datetime.timedelta(days=5 * 365)
max_date = datetime.datetime.now()
selected_dates = st.date_input(
    "Select Range of Dates for Stock Data",
    (min_date, max_date),
    min_date,
    max_date,
    format="MM/DD/YYYY"
)
st.caption('The max date will be used for stock selection')

# Get Polygon API data
file_path = os.path.join(os.getcwd().replace('/pages', ''), 'keys.json')
with open(file_path, 'r') as file:
    keys_dict = json.load(file)
polygon_api_key = keys_dict['POLYGON_API_KEY']
stock_query = utils._get_daily_aggregates(api_key=polygon_api_key, adjusted='true', date=selected_dates[1].strftime('%Y-%m-%d'))
stock_symbols = [result['T'] for result in stock_query['results']]
stock_symbols = sorted(stock_symbols)

# Choose stock symbol
selected_stock = st.selectbox('Select a stock symbol', stock_symbols)
st.divider()

# Retrieve stock data
stock_data = utils._get_ticker_aggregates(
    min_date=selected_dates[0].strftime('%Y-%m-%d'),
    max_date=selected_dates[1].strftime('%Y-%m-%d'),
    stock_symbol=selected_stock, 
    api_key=polygon_api_key,
    adjusted='true'
)
st.json(stock_data)