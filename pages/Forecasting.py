import os
import sys
import json
import utils
import prophet
import datetime
import requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

st.title('Forecasting')
st.markdown(
    """
    This page covers forecasting of time series data (in this case stock data) using the Prophet library from Meta.
    Other forecasting methods are available, an example deep learning model is available to use but requires retraining to remain relevant.
    """
)

# Get date
min_date = datetime.datetime.now() - datetime.timedelta(days=5 * 365)
selected_date = st.date_input(label='Select a date', min_value=min_date, max_value=datetime.datetime.now())
st.text(f'The selected date {selected_date} will be used to forecast from')

# Get Polygon API data
file_path = os.path.join(os.getcwd().replace('/pages', ''), 'keys.json')
with open(file_path, 'r') as file:
    keys_dict = json.load(file)
polygon_api_key = keys_dict['POLYGON_API_KEY']
stock_query = utils._get_daily_aggregates(api_key=polygon_api_key, adjusted='true', date=selected_date.strftime('%Y-%m-%d'))
stock_symbols = [result['T'] for result in stock_query['results']]
stock_symbols = sorted(stock_symbols)

# Choose stock symbol
stock_symbol = st.selectbox('Select a stock symbol', stock_symbols)
st.divider()

# Display stock forecast and trends
