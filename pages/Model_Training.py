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

st.title('Forecast Model Training')

