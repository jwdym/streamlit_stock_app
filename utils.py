import os
import datetime
import prophet
import plotly
import sklearn
import json
import requests
import random
import re
import mlflow
import utils

import numpy as np
import pandas as pd
import multiprocessing as mp
import plotly.graph_objects as go

from prophet import Prophet
from streamlit.components.v1 import html
from mlflow.models import infer_signature
from prophet.diagnostics import cross_validation, performance_metrics
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
        date (str): The date to get stock data for
        api_key (str): The API key to use
        adjusted (str): The adjusted flag for the API
    Outputs:
        response (dict): The stock data response
    """
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    base_url = 'https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/'
    response = requests.get(f'{base_url}{date}?adjusted={adjusted}&apiKey={api_key}')
    return(response.json())

def _get_ticker_aggregates(min_date:str, max_date:str, stock_symbol:str, api_key:str, adjusted:str='true'):
    """
    Function to get ticker aggregates of stock data
    Inputs:
        min_date (str): The minimum date to get stock data for
        max_date (str): The maximum date to get stock data for
        stock_symbol (str): The stock symbol to get data for
        api_key (str): The API key to use
        adjusted (str): The adjusted flag for the API
    Outputs:
        response (dict): The stock data response
    """
    if min_date is None:
        min_date = datetime.datetime.now() - datetime.timedelta(days=5 * 365)
    if max_date is None:
        max_date = datetime.datetime.now()
    base_url = f'https://api.polygon.io/v2/aggs/ticker/'
    response = requests.get(f'{base_url}{stock_symbol}/range/1/day/{min_date}/{max_date}?adjusted={adjusted}&sort=asc&apiKey={api_key}')
    return(response.json())

def _get_ticker_details(date:str, stock_symbol:str, api_key:str):
    """
    Function to retreive stock details
    Inputs:
        date (str): The date to get stock data for
        stock_symbol (str): The stock symbol to get data for
        api_key (str): The API key to use
    Outputs:
        response (dict): The stock data response
    """
    base_url = 'https://api.polygon.io/v3/reference/tickers/'
    response = requests.get(f'{base_url}{stock_symbol}?date={date}&apiKey={api_key}')
    return(response.json())

def update_stock_data(min_date:str, max_date:str, api_key:str):
    """
    Function to update stock data
    Inputs:
        min_date (str): The minimum date to get stock data for
        max_date (str): The maximum date to get stock data for
        api_key (str): The API key to use
    Outputs:
        None
    """
    # Get polygon data from each date in list
    counter = 1
    min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d')
    date_list = [(min_date + datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (max_date-min_date).days)]

    # Iterate through dates
    for date in date_list:
        try:
            daily_report = _get_daily_aggregates(
                date=date,
                api_key=api_key,
                adjusted='True'
            )
            with open(f'data/daily-aggregates/{date.replace("-", "")}.json', 'w') as outfile:
                json.dump(daily_report, outfile)
        except Exception as e:
            print(e)

        if counter % 5 == 0:
            print("{:.4f} %".format(100 * counter/len(date_list)))
        counter += 1

def get_max_date(file_dir:str):
    """
    Function to get the max date from a directory of files
    Inputs:
        file_dir (str): The directory of files
    Outputs:
        max(file_dates) (datetime): The max date from the files
    """
    file_names = os.listdir(file_dir)
    file_dates = []
    for file_name in file_names:
        file_dates.append(datetime.datetime.strptime(file_name.split('.')[0], '%Y%m%d'))
    return(max(file_dates))

def get_min_date(file_dir:str):
    """
    Function to get the min date from a directory of files
    Inputs:
        file_dir (str): The directory of files
    Outputs:
        min(file_dates) (datetime): The min date from the files
    """
    file_names = os.listdir(file_dir)
    file_dates = []
    for file_name in file_names:
        file_dates.append(datetime.datetime.strptime(file_name.split('.')[0], '%Y%m%d'))
    return(min(file_dates))

def create_stock_dataframe(file_dir:str):
    """
    Function to create a stock dataframe from a directory of files
    Inputs:
        file_dir (str): The directory of files
    Outputs:
        stock_json (Pandas Dataframe): The stock data as a dataframe
    """

    stock_json = {
        'Date': [],
        'Exchange_Symbol': [],
        'Close_Price': [],
        'Highest_Price': [],
        'Lowest_Price': [],
        'Transactions': [],
        'Open_Price': [],
        'Timestamp': [],
        'Trading_Volume': [],
        'Volume_Weighted_AVG_Price': []
    }

    # Iterate through files and append data to dict
    for file_name in os.listdir(file_dir):
        file_date = datetime.datetime.strptime(file_name.split('.')[0], '%Y%m%d')
        with open(f'{file_dir}/{file_name}') as f:
            temp_json = json.load(f)

        if temp_json['queryCount'] == 0:
            stock_json['Date'].append(file_date.strftime('%Y-%m-%d'))
            stock_json['Exchange_Symbol'].append(None)
            stock_json['Close_Price'].append(None)
            stock_json['Highest_Price'].append(None)
            stock_json['Lowest_Price'].append(None)
            stock_json['Transactions'].append(None)
            stock_json['Open_Price'].append(None)
            stock_json['Timestamp'].append(None)
            stock_json['Trading_Volume'].append(None)
            stock_json['Volume_Weighted_AVG_Price'].append(None)

        else:
            # Add stock data to json file
            for stock_file in temp_json['results']:
                stock_json['Date'].append(file_date.strftime('%Y-%m-%d'))
                stock_json['Exchange_Symbol'].append(stock_file['T'])
                stock_json['Close_Price'].append(stock_file['c'])
                stock_json['Highest_Price'].append(stock_file['h'])
                stock_json['Lowest_Price'].append(stock_file['l'])
                try:
                    stock_json['Transactions'].append(stock_file['n'])
                except:
                    stock_json['Transactions'].append(None)
                stock_json['Open_Price'].append(stock_file['o'])
                stock_json['Timestamp'].append(stock_file['t'])
                stock_json['Trading_Volume'].append(stock_file['v'])
                try:
                    stock_json['Volume_Weighted_AVG_Price'].append(stock_file['vw'])
                except:
                    stock_json['Volume_Weighted_AVG_Price'].append(None)

    return(pd.DataFrame(stock_json))

def create_forecast_params(param_dict):
    print(f'Creating forecast parameters for {param_dict["stock_symbol"]}...')
    return {
        'forecast_metric': 'Close_Price',
        'forecast_window': 30,
        'stock_symbol': param_dict['stock_symbol'],
        'df': param_dict['df'].loc[param_dict['df'].Exchange_Symbol == param_dict['stock_symbol']]
    }

def train_model(params):
    # Get min and max date for data frame
    date_sequence = pd.date_range(start=params['df'].Date.min(), end=params['df'].Date.max())
    date_df = pd.DataFrame({'Date': date_sequence})
    temp_df = date_df.merge(params['df'], on='Date', how='left')
    temp_df.fillna(method='ffill', inplace=True)

    # Create a training, testing, and validation dataframe based on a forecasting window
    # train_df = temp_df.loc[temp_df.Date < temp_df.Date.max() - pd.Timedelta(days=params['forecast_window'] * 2)]
    # test_df = temp_df.loc[temp_df.Date >= temp_df.Date.max() - pd.Timedelta(days=params['forecast_window'])]
    # val_df = temp_df.loc[(temp_df.Date >= temp_df.Date.max() - pd.Timedelta(days=params['forecast_window'] * 2)) & (temp_df.Date < temp_df.Date.max() - pd.Timedelta(days=params['forecast_window']))]

    # Rename columns for modeling
    train_df = temp_df[['Date', params['forecast_metric']]].rename(columns={"Date": "ds", params['forecast_metric']: "y"})
    # train_df = train_df[['Date', params['forecast_metric']]].rename(columns={"Date": "ds", params['forecast_metric']: "y"})
    # test_df = test_df[['Date', params['forecast_metric']]].rename(columns={"Date": "ds", params['forecast_metric']: "y"})
    # val_df = val_df[['Date', params['forecast_metric']]].rename(columns={"Date": "ds", params['forecast_metric']: "y"})

    # Prophet model
    prophet_model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        seasonality_mode='multiplicative'
    )
    prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=4)
    prophet_model.add_seasonality('quarterly', period=91.25, fourier_order=8)
    print(f'Fitting model for {params["stock_symbol"]}...')
    with mlflow.start_run(nested=True, run_name=f"{params['stock_symbol']}-{datetime.datetime.strftime(train_df.ds.max(), '%Y%m%d')}") as child_run:
        prophet_model.fit(train_df)

        # extract and log parameters such as changepoint_prior_scale in the mlflow run
        model_params = {
            name: value for name, value in vars(prophet_model).items() if np.isscalar(value)
        }
        mlflow.log_params(model_params)

        # Cross validation
        # initial_days = int(train_df.shape[0] * 0.9) # get 90% of available data for initial period
        initial_days = 900 # default at 900 for now
        cv_results = cross_validation(
            prophet_model, initial=f"{initial_days} days", period=f"{params['forecast_window']} days", horizon=f"{params['forecast_window']} days"
        )

        # Calculate metrics from cv_results, then average each metric across all backtesting windows and log to mlflow
        cv_metrics = ["mse", "rmse", "mape"]
        metrics_results = performance_metrics(cv_results, metrics=cv_metrics)
        average_metrics = metrics_results.loc[:, cv_metrics].mean(axis=0).to_dict()
        mlflow.log_metrics(average_metrics)

        # Calculate model signature
        train = prophet_model.history
        predictions = prophet_model.predict(prophet_model.make_future_dataframe(params['forecast_window']))
        signature = infer_signature(train, predictions)

        model_info = mlflow.prophet.log_model(
            prophet_model, "prophet-model", signature=signature
        )
    print(f'Finished modeling for {params["stock_symbol"]}...')