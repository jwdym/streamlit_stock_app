import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from PIL import Image
from utils import nav_page

st.set_page_config(
    page_title="Tate Investment Managment",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Introduction
st.header("Investment Management Dashboard")
forecast_image = Image.open('static/forecast_image.png')
st.image(forecast_image, width=500)
st.markdown(
    """
    Welcome to the Tate Investment Management dashboard! This dashboard was created to help manage investments.
    """
)
if st.button("Forecasting"):
    nav_page("Forecasting")