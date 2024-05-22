import os
import prophet
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from utils import stock_data_forecast, plot_forecast_vs_actual, regression_scores

st.title('Forecasting')
st.markdown(
    """
    This page covers forecasting of time series data (in this case stock data) using the Prophet library from Meta.
    Other forecasting methods are available, an example deep learning model is available to use but requires retraining to remain relevant.
    """
)
