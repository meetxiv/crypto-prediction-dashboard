# Cryptocurrency Prediction Dashboard

An interactive dashboard for cryptocurrency price analysis and prediction using machine learning models.

## Features

- Historical price data visualization
- Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- Price predictions using multiple ML models:
  - Linear Regression
  - Random Forest
  - Neural Network
- Comparative cryptocurrency performance analysis

## Data Source

This application uses Yahoo Finance API (via yfinance library) to fetch real-time cryptocurrency data.

## Libraries Used

- Streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- plotly

## Getting Started

1. Install the required packages: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py` 