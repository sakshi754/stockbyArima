import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Streamlit App Title
st.title("Stock Price Prediction using ARIMA Model")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")  # Fetch last 1 year of data
    return df

if st.button("Predict"):
    # Fetch stock data
    df = fetch_stock_data(ticker)
    
    if df.empty:
        st.error("Invalid ticker or no data available. Please try again.")
    else:
        df.reset_index(inplace=True)
        df['Days'] = np.arange(len(df))

        # Prepare data for ARIMA
        prices = df['Close'].values
        
        # Fit ARIMA model (using ARIMA(5,1,0) as an initial configuration)
        model = ARIMA(prices, order=(5,1,0))  # (p,d,q) - tweak as needed
        model_fit = model.fit()

        # Predict the next 30 days
        future_forecast = model_fit.forecast(steps=30)
        
        # Generate future dates
        future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=31, freq='D')[1:]

        # Plot Predictions
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['Close'], label="Historical Prices", color="blue")
        plt.plot(future_dates, future_forecast, label="Predicted Prices", linestyle='dashed', color="red")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Stock Price Prediction for {ticker} (Next 30 Days)")
        plt.legend()

        # Display plot in Streamlit
        st.pyplot(plt)
