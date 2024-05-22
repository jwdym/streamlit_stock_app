import os
import datetime
import prophet
import plotly
import sklearn
import requests
import random
import re

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from prophet import Prophet
from streamlit.components.v1 import html
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def stock_data_forecast(df: pd.DataFrame, date_col: str, forecast_metric: str, forecast_window: int):
    """
    Function to get the stock forecast for a specific stock
    Inputs:
        df (pd dataframe): a pandas dataframe of stock data
        stock_symbol (str): the specific stock symbol to predict for
        forecast_metric (str): the stock data to forecast on
        forecast_window (int): the look forward prediction window
    Output:
        forecast (pd dataframe): a pandas dataframe of stock forecast
    """

    # get date sequence to fill in gaps
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    date_sequence = pd.date_range(start=min_date,end=max_date)
    date_df = pd.DataFrame({'Date': date_sequence})

    # join data frames
    temp_df = date_df.merge(df, on='Date', how='left')
    # fill in weekend and holidays with last value
    temp_df.fillna(method='ffill', inplace=True)

    # generate forecast
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=4)
    m.add_seasonality('quarterly', period=91.25, fourier_order=8)
    m.add_country_holidays(country_name='US')
    model_df = temp_df[['Date', forecast_metric]].rename(columns={"Date": "ds", forecast_metric: "y"})
    m.fit(model_df)
    future = m.make_future_dataframe(periods=forecast_window)
    forecast = m.predict(future)

    # return forecast
    return(m, forecast)

def regression_scores(df: pd.DataFrame, y_true: str, y_pred: str):
    """
    Function to get regression evaluation scores for a model
    Inputs:
        df (Pandas Dataframe): The dataframe with obs and predictions
        y_true (str): The observed metric
        y_pred (str): The predicted metric
    outputs:
        model_scores (dict): The model evaluation scores
    """

    # Get model scores
    model_scores = {
        'explained_variance_score': explained_variance_score(df[y_true], df[y_pred]),
        'max_error': max_error(df[y_true], df[y_pred]),
        'mean_absolute_error': mean_absolute_error(df[y_true], df[y_pred]),
        'mean_squared_error': mean_squared_error(df[y_true], df[y_pred]),
        'r2_score': r2_score(df[y_true], df[y_pred]),
    }

    return(model_scores)

def plot_forecast_vs_actual(forecast_df: pd.DataFrame, df: pd.DataFrame, forecast_metric: str, date_col: str):
    """
    Function to plot the forecasted stock values vs observed
    Inputs:
        forecast_df (Pandas Dataframe): Forecast data set
        df (Pandas Dataframe): The holdout data set
        forecast_metric (str): The metric being forecast
        date_col (str): The date column of the data sets
    outputs:
        fig: A plotly graph object
        model_scores (dict): The model evaluation scores
    """
    # Get data
    temp_df = df.merge(forecast_df, how='inner', left_on='Date', right_on='ds')
    temp_df = temp_df[['Date', forecast_metric, 'yhat_lower', 'yhat', 'yhat_upper']]

    # Create plots
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.yhat_upper, connectgaps=True, name='Prediction Upper'))
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.yhat_lower, connectgaps=True, name='Prediction Lower', fill='tonexty'))
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.yhat, connectgaps=True, name='Prediction'))
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.Close_Price, connectgaps=True, name='Close Price'))

    fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

    return(fig)

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

def _get_daily_aggregates(date:str, api_key:str, adjusted:str='true'):
    """
    Function to get daily aggregates of stock data
    Inputs:
        
    Outputs:
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    base_url = 'https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/'
    response = requests.get(f'{base_url}{date}?adjusted={adjusted}&apiKey={api_key}')
    return(response.json())