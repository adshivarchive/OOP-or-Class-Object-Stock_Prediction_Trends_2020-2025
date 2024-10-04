# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:46:47 2024

@author: LENOVO
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = LinearRegression()
        
    def fetch_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data['dates_numeric'] = self.data.index.map(pd.Timestamp.timestamp)
        
    def plot_data(self):
        plt.figure(figsize=(15,6))
        plt.plot(self.data.index, self.data["Adj Close"], label="Actual Price", color="purple")
        plt.grid(linestyle=":")
        plt.ylabel("Price ($)")
        plt.title(f"Stock Price from {self.start_date} to {self.end_date}")
       
    def train_and_predict(self):
        x = self.data['dates_numeric'].values.reshape(-1, 1)
        y = self.data['Adj Close']
        
        self.model.fit(x, y)
        
        y_pred = self.model.predict(x)
        plt.plot(self.data.index, y_pred, label="Interpolation", color="gold", linestyle="-")
        
        future_dates = pd.date_range(start=self.data.index[-1], periods=365, freq='D')
        future_dates_numeric = future_dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)
        future_pred = self.model.predict(future_dates_numeric)
        
        plt.plot(future_dates, future_pred, label="Prediction (2025)", color="magenta", linestyle="--")
        
    def show_plot(self):
        plt.legend()
        plt.savefig(f"{self.ticker}_Stock_Price.png", dpi=300)
        plt.show()
        
predictor = StockPredictor("AAPL", "2020-01-01","2024-09-25")
predictor.fetch_data()
predictor.plot_data()
predictor.train_and_predict()
predictor.show_plot()