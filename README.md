# Time Series Analysis

### What is Time Series Analysis

### Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values. While regression analysis is often employed in such a way as to test relationships between one or more different time series, this type of analysis is not usually called "time series analysis", which refers in particular to relationships between different points in time within a single series. Interrupted time series analysis is used to detect changes in the evolution of a time series from before to after some intervention which may affect the underlying variable.

# What are they steps for " Time Series Anlysis"

Data Source link is https://finance.yahoo.com/ .

      Once we taken from the data we install the standard libraries such as given below

      import pandas as pd
      import matplotlib.pyplot as plt
      import numpy as np
      from mplfinance.original_flavor import candlestick_ohlc
      import matplotlib.dates as mpdates
      plt.style.use('dark_background')

Then move to the data loading pandas

       data=pd.read_csv('filename.csv')

Then visualize the
