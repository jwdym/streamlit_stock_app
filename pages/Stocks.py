import os
import sys
import json
import utils
import datetime
import requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

#################
# Stock details #
#################
st.subheader('Stock Details')
stock_details = utils._get_ticker_details(
    date=selected_dates[1].strftime('%Y-%m-%d'),
    stock_symbol=selected_stock,
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

st.divider()

##############
# Stock data #
##############

# Retrieve stock data
stock_data = utils._get_ticker_aggregates(
    min_date=selected_dates[0].strftime('%Y-%m-%d'),
    max_date=selected_dates[1].strftime('%Y-%m-%d'),
    stock_symbol=selected_stock, 
    api_key=polygon_api_key,
    adjusted='true'
)
stock_json = {
    'Date': [],
    'Exchange_Symbol': [],
    'Close_Price': [],
    'Highest_Price': [],
    'Lowest_Price': [],
    'Transactions': [],
    'Open_Price': [],
    'Trading_Volume': [],
    'Volume_Weighted_AVG_Price': []
}

for result in stock_data['results']:
    stock_json['Date'].append(datetime.datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'))
    stock_json['Exchange_Symbol'].append(selected_stock)
    stock_json['Close_Price'].append(result['c'])
    stock_json['Highest_Price'].append(result['h'])
    stock_json['Lowest_Price'].append(result['l'])
    stock_json['Transactions'].append(result['n'])
    stock_json['Open_Price'].append(result['o'])
    stock_json['Trading_Volume'].append(result['v'])
    stock_json['Volume_Weighted_AVG_Price'].append(result['vw'])

# Display stock data
st.subheader('Stock Data')
with st.expander("Show Stock Data"):
    stock_df = pd.DataFrame(stock_json)
    st.dataframe(stock_df)

# Plot price data
fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text=f"'{selected_stock}' Price Data")
    )
)
for metric in ['Close_Price', 'Highest_Price', 'Lowest_Price', 'Open_Price', 'Volume_Weighted_AVG_Price']:
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df[metric],
        mode='lines',
        name=metric
        )
    )
st.plotly_chart(fig, theme='streamlit')

# Plot volume data
fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text=f"'{selected_stock}' Volume Data")
    )
)
for metric in ['Trading_Volume', 'Transactions']:
    fig.add_trace(go.Bar(
        x=stock_df['Date'],
        y=stock_df[metric],
        name=metric
        )
    )
st.plotly_chart(fig, theme='streamlit')