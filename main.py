import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# List of ETFs
etfs = ['SUSW.MI', 'IESE.AS', 'ELCR.PA', 'GRON.F', 'ICLN', 'AYEM.DE', 'DJRE.AX']

def weighted_moving_average(df, window):
    weights = np.arange(1, window + 1)
    wma = df.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

def app():
    """
    ETF Analyzer and Portfolio Optimizer

    The ETF Analyzer and Portfolio Optimizer is a Streamlit-based web application designed for financial enthusiasts, analysts, and investors who want to analyze and optimize their ETF portfolios.
    This app provides users with tools to perform in-depth analysis of selected ETFs by fetching historical data from Yahoo Finance, visualizing price trends, and calculating key moving averages.

    Key Features:
    1. ETF Selection: Users can select multiple ETFs from a predefined list for analysis.
    2. Data Download: Fetches historical adjusted close price data for the selected ETFs from Yahoo Finance.
    3. Data Cleaning: Handles missing data by forward-filling and back-filling techniques to ensure continuity.
    4. Raw Data Display: Option to display the raw data for the selected ETFs.
    5. Price Visualization: Plots the adjusted close prices for the selected ETFs over time.
    6. Moving Averages: Calculates and plots short-term and long-term moving averages for each ETF.
    7. Weighted Moving Averages: Calculates and plots weighted moving averages, allowing users to set the window size.
    8. Interactive Elements: Sliders to adjust the window sizes for moving averages dynamically.
    """

    # Title and selection
    st.title('ETF Analyzer and Portfolio Optimizer')
    selected_etfs = st.multiselect('Select ETFs for Analysis:', etfs, default=etfs)

    # Data download
    data = yf.download(selected_etfs, start="2020-01-01", end="2024-05-05")
    cleaned_data = {}
    for etf in selected_etfs:
        df = data['Adj Close'][etf].copy() if len(selected_etfs) > 1 else data['Adj Close'].copy()
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        cleaned_data[etf] = df

    # Data exploration (optional)
    if st.checkbox('Show Raw Data'):
        st.subheader('Raw Data')
        for etf, data in cleaned_data.items():
            st.write(f'**{etf}**')
            st.dataframe(data.tail())

    # Plot adjusted close prices
    st.subheader('Adjusted Close Prices')
    fig, ax = plt.subplots(figsize=(14, 8))
    for etf, data in cleaned_data.items():
        ax.plot(data.index, data, label=etf)
    ax.set_title('Adjusted Close Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Moving Averages
    st.subheader('Moving Averages')
    window1 = st.slider('Short-term moving average window:', 100, 250, 150)
    window2 = st.slider('Long-term moving average window:', window1 + 1, 500, 200)

    for etf, data in cleaned_data.items():
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(data, label='Adjusted Close')
        ma1 = data.rolling(window=window1).mean()
        ma2 = data.rolling(window=window2).mean()
        ax.plot(ma1, label=f'{window1}-day MA')
        ax.plot(ma2, label=f'{window2}-day MA')
        ax.set_title(f'{etf} Moving Averages')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

    # Weighted Moving Average
    st.subheader('Weighted Moving Average')
    window = st.slider('Weighted moving average window:', 50, 200, 100)

    for etf, data in cleaned_data.items():
        wma = weighted_moving_average(data, window)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(data, label='Adjusted Close')
        ax.plot(wma, label=f'{window}-day WMA')
        ax.set_title(f'{etf} Weighted Moving Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    app()
