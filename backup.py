import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(
    page_title="Crypto Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)


# Sidebar for navigation
st.sidebar.title("Crypto Prediction Dashboard")
st.sidebar.markdown("---")


# Define crypto options
crypto_options = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Dogecoin': 'DOGE-USD',
    'Cardano': 'ADA-USD',
    'Solana': 'SOL-USD',
    'Litecoin': 'LTC-USD',
    'Ripple': 'XRP-USD',
    'Polkadot': 'DOT-USD'
}

# User inputs
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
crypto_ticker = crypto_options[selected_crypto]

period_options = {'1 Month': '1mo', '3 Months': '3mo', '6 Months': '6mo', '1 Year': '1y', '2 Years': '2y', '5 Years': '5y', 'Max': 'max'}
selected_period = st.sidebar.selectbox("Select Historical Data Period", list(period_options.keys()))
period = period_options[selected_period]

prediction_days = st.sidebar.slider("Number of Days to Predict", 7, 90, 30)

model_options = ['Linear Regression', 'Random Forest', 'LSTM Neural Network']
selected_model = st.sidebar.selectbox("Select Prediction Model", model_options)

# Function to load crypto data
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        
        # Handle multi-index columns properly
        if isinstance(data.columns, pd.MultiIndex):
            # This handles the common case where Yahoo returns multi-index columns
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        
        # Check if we have the expected columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns and f'Adj {col}' in data.columns:
                data[col] = data[f'Adj {col}']
            elif col not in data.columns and f'{col}_' in [c[:len(col)+1] for c in data.columns]:
                # Find the column that starts with the required name
                matching_col = [c for c in data.columns if c.startswith(f'{col}_')][0]
                data[col] = data[matching_col]
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Fetch data
data = load_data(crypto_ticker, period)

# Check if data is empty or doesn't have required columns
if data.empty:
    st.error("No data available for the selected cryptocurrency and time period.")
    st.stop()

# Ensure we have the necessary columns
required_columns = ['Open', 'High', 'Low', 'Close']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    # Debug info for column issues
    st.error(f"Missing required columns: {missing_columns}. Available columns: {data.columns.tolist()}")
    st.stop()

# Main content area
st.title(f"{selected_crypto} Price Prediction Dashboard")

# Show dataset info
st.subheader("Historical Data Overview")
col1, col2, col3, col4 = st.columns(4)

# Current price - Convert Series to float before formatting
current_price = float(data['Close'].iloc[-1])
col1.metric("Current Price", f"${current_price:.2f}")

# Price change (1 day) - Also convert to float
if len(data) > 1:
    price_change_1d = float(data['Close'].iloc[-1] - data['Close'].iloc[-2])
    price_change_percent_1d = (price_change_1d / float(data['Close'].iloc[-2])) * 100
    col2.metric("24h Change", f"${price_change_1d:.2f}", f"{price_change_percent_1d:.2f}%")
else:
    col2.metric("24h Change", "Insufficient data")

# Price change (7 days) - Convert to float
if len(data) > 7:
    price_change_7d = float(data['Close'].iloc[-1] - data['Close'].iloc[-7])
    price_change_percent_7d = (price_change_7d / float(data['Close'].iloc[-7])) * 100
    col3.metric("7d Change", f"${price_change_7d:.2f}", f"{price_change_percent_7d:.2f}%")
else:
    col3.metric("7d Change", "Insufficient data")

# Trading volume - Convert to float
try:
    if 'Volume' in data.columns:
        volume = float(data['Volume'].iloc[-1])
        col4.metric("Trading Volume", f"${volume:,.0f}")
    else:
        col4.metric("Trading Volume", "N/A")
except:
    col4.metric("Trading Volume", "N/A")

# Historical price chart
st.subheader("Historical Price Chart")
tab1, tab2 = st.tabs(["Line Chart", "Candlestick Chart"])

with tab1:
    # Use go.Figure instead of px.line to avoid multi-index issues
    fig = go.Figure()
    
    # Add traces directly from dataframe columns
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines',
        name='Close'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Open'], 
        mode='lines',
        name='Open'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['High'], 
        mode='lines',
        name='High'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Low'], 
        mode='lines',
        name='Low'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{selected_crypto} Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # For candlestick, explicitly ensure data types are correct
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))
    
    fig.update_layout(
        title=f"{selected_crypto} Candlestick Chart", 
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

# Volume chart - also switch to go.Figure
st.subheader("Trading Volume")
volume_fig = go.Figure()
try:
    if 'Volume' in data.columns:
        volume_fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume'
        ))
        volume_fig.update_layout(
            title=f"{selected_crypto} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume"
        )
        st.plotly_chart(volume_fig, use_container_width=True)
    else:
        st.warning("Volume data is not available for this cryptocurrency.")
except Exception as e:
    st.error(f"Unable to display volume data: {e}")

# Price prediction
st.subheader(f"{selected_crypto} Price Prediction for Next {prediction_days} Days")
st.info("Please note that cryptocurrency price predictions are speculative and should not be used as financial advice.")

# Add a train model button
train_button = st.button("Train New Model")

# Prepare data for prediction
df = data.reset_index()[['Date', 'Close']]
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Function to create features
def create_features(data, window_size=14):
    # Create features using rolling window
    df_features = data.copy()
    # Rolling mean
    df_features['rolling_mean'] = df_features['y'].rolling(window=window_size).mean()
    # Rolling standard deviation
    df_features['rolling_std'] = df_features['y'].rolling(window=window_size).std()
    # Price momentum
    df_features['momentum'] = df_features['y'] - df_features['y'].shift(window_size)
    # Fill NaN values
    df_features = df_features.fillna(method='bfill')
    return df_features

# Feature engineering
feature_df = create_features(df)

# Predict function
def predict_prices(model_name, data, prediction_days):
    # Split data for training
    train_data = data.copy()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data[['y', 'rolling_mean', 'rolling_std', 'momentum']])
    
    # Define features and target
    X = scaled_data
    y = train_data['y'].values
    
    # Get last known values for prediction
    last_values = scaled_data[-1].reshape(1, -1)
    
    # Create prediction dates
    last_date = train_data['ds'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
    
    # Initialize predictions
    predictions = []
    
    if model_name == 'Linear Regression':
        # Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions iteratively
        current_prediction = last_values
        for _ in range(prediction_days):
            pred = model.predict(current_prediction)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            # We need to recalculate the features based on the new prediction
            new_row = np.copy(current_prediction[0])
            new_row[0] = pred  # Update closing price
            # (Simplified feature update)
            current_prediction = np.array([new_row])
            
    elif model_name == 'Random Forest':
        # Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Make predictions iteratively
        current_prediction = last_values
        for _ in range(prediction_days):
            pred = model.predict(current_prediction)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            new_row = np.copy(current_prediction[0])
            new_row[0] = pred
            current_prediction = np.array([new_row])
            
    elif model_name == 'LSTM Neural Network':
        # Use MLP Regressor as an alternative to LSTM
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50), 
            activation='relu', 
            solver='adam', 
            max_iter=500,
            random_state=42
        )
        model.fit(X, y)
        
        # Make predictions iteratively
        current_prediction = last_values
        for _ in range(prediction_days):
            pred = model.predict(current_prediction)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            new_row = np.copy(current_prediction[0])
            new_row[0] = pred
            current_prediction = np.array([new_row]).reshape(1, -1)
    
    # Create prediction dataframe
    future_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions
    })
    
    return future_df

# Make prediction based on selected model
if train_button or 'future' not in st.session_state:
    with st.spinner(f"Running {selected_model} model..."):
        future = predict_prices(selected_model, feature_df, prediction_days)
        st.session_state.future = future
else:
    future = st.session_state.future

# Display prediction chart
fig = go.Figure()
# Historical data
fig.add_trace(go.Scatter(
    x=df['ds'],
    y=df['y'],
    mode='lines',
    name='Historical',
    line=dict(color='blue')
))
# Prediction
fig.add_trace(go.Scatter(
    x=future['ds'],
    y=future['yhat'],
    mode='lines',
    name='Prediction',
    line=dict(color='red')
))
fig.update_layout(
    title=f"{selected_crypto} Price Prediction ({selected_model})",
    xaxis_title="Date",
    yaxis_title="Price (USD)"
)
st.plotly_chart(fig, use_container_width=True)

# Display prediction data table
st.subheader("Predicted Prices")

# Fix the formatting issue by explicitly converting to Python float type
formatted_future = future.copy()
formatted_future['Predicted Price'] = formatted_future['yhat'].apply(lambda x: f"${float(x):.2f}")
formatted_future = formatted_future.rename(columns={'ds': 'Date'}).drop(columns=['yhat'])

st.dataframe(formatted_future, use_container_width=True)

# Technical analysis indicators section
st.subheader("Technical Analysis Indicators")

indicator_tab1, indicator_tab2, indicator_tab3 = st.tabs(["Moving Averages", "RSI & MACD", "Bollinger Bands"])

with indicator_tab1:
    # Create a copy of the data to avoid affecting the original data
    ma_data = data.copy()
    
    # Create a properly structured dataframe for moving averages
    ma_df = pd.DataFrame(index=ma_data.index)
    ma_df['Close'] = ma_data['Close']
    ma_df['MA5'] = ma_data['Close'].rolling(window=5).mean()
    ma_df['MA20'] = ma_data['Close'].rolling(window=20).mean()
    ma_df['MA50'] = ma_data['Close'].rolling(window=50).mean()
    
    # Use go.Figure instead of px.line
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df['MA5'], mode='lines', name='MA5'))
    fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df['MA20'], mode='lines', name='MA20'))
    fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df['MA50'], mode='lines', name='MA50'))
    
    fig.update_layout(
        title='Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with indicator_tab2:
    col1, col2 = st.columns(2)
    
    # Create separate dataframes for RSI and MACD
    rsi_data = data.copy()
    close_series = rsi_data['Close']
    
    # Calculate RSI
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # RSI Chart - Use go.Figure
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi, mode='lines', name='RSI'))
    
    # Add horizontal lines for overbought/oversold levels
    fig_rsi.add_shape(
        type="line", line=dict(dash="dash", color="red"),
        x0=rsi_data.index[0], x1=rsi_data.index[-1], y0=70, y1=70
    )
    fig_rsi.add_shape(
        type="line", line=dict(dash="dash", color="green"),
        x0=rsi_data.index[0], x1=rsi_data.index[-1], y0=30, y1=30
    )
    
    fig_rsi.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100])
    )
    
    col1.plotly_chart(fig_rsi, use_container_width=True)
    
    # Calculate MACD
    macd_data = data.copy()
    close_series = macd_data['Close']
        
    # Calculate MACD components
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # MACD Chart - Use go.Figure
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=macd_data.index, y=macd_line, mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=macd_data.index, y=signal_line, mode='lines', name='Signal'))
    
    # Add MACD histogram
    histogram = macd_line - signal_line
    colors = ['red' if val < 0 else 'green' for val in histogram]
    fig_macd.add_trace(go.Bar(x=macd_data.index, y=histogram, name='Histogram', marker_color=colors))
    
    fig_macd.update_layout(
        title='Moving Average Convergence Divergence (MACD)',
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    col2.plotly_chart(fig_macd, use_container_width=True)

with indicator_tab3:
    # Create a new dataframe for Bollinger Bands
    bb_data = data.copy()
    
    # Create a properly structured dataframe
    bb_df = pd.DataFrame(index=bb_data.index)
    bb_df['Close'] = bb_data['Close']
    bb_df['MA20'] = bb_data['Close'].rolling(window=20).mean()
    bb_df['Upper'] = bb_df['MA20'] + (bb_data['Close'].rolling(window=20).std() * 2)
    bb_df['Lower'] = bb_df['MA20'] - (bb_data['Close'].rolling(window=20).std() * 2)
    
    # Bollinger Bands Chart
    fig_bb = go.Figure()
    
    # Add fill between upper and lower bands
    fig_bb.add_trace(go.Scatter(
        x=bb_df.index, 
        y=bb_df['Upper'],
        name='Upper Band',
        line=dict(color='rgba(173, 204, 255, 0.7)'),
        mode='lines'
    ))
    
    fig_bb.add_trace(go.Scatter(
        x=bb_df.index,
        y=bb_df['Lower'],
        name='Lower Band',
        fill='tonexty',
        fillcolor='rgba(173, 204, 255, 0.2)',
        line=dict(color='rgba(173, 204, 255, 0.7)'),
        mode='lines'
    ))
    
    fig_bb.add_trace(go.Scatter(
        x=bb_df.index, 
        y=bb_df['MA20'], 
        name='MA20', 
        line=dict(color='blue', width=1)
    ))
    
    fig_bb.add_trace(go.Scatter(
        x=bb_df.index, 
        y=bb_df['Close'], 
        name='Close', 
        line=dict(color='black', width=1)
    ))
    
    fig_bb.update_layout(
        title='Bollinger Bands', 
        xaxis_title='Date', 
        yaxis_title='Price',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_bb, use_container_width=True)

# Market comparison section
st.subheader("Market Comparison")

comparison_cryptos = st.multiselect(
    "Compare with other cryptocurrencies",
    list(crypto_options.keys()),
    default=[list(crypto_options.keys())[1] if selected_crypto != list(crypto_options.keys())[1] else list(crypto_options.keys())[0]]
)

if comparison_cryptos:
    with st.spinner("Loading comparison data..."):
        # Get data for comparison
        comparison_data = {}
        for crypto in comparison_cryptos:
            ticker = crypto_options[crypto]
            comp_data = yf.download(ticker, period=period)
            # Handle multi-index columns if present
            if isinstance(comp_data.columns, pd.MultiIndex):
                comp_data.columns = ['_'.join(col).strip() for col in comp_data.columns.values]
                
            # Ensure we have the 'Close' column
            if 'Close' not in comp_data.columns:
                if 'Adj_Close' in comp_data.columns:
                    comp_data['Close'] = comp_data['Adj_Close']
                elif 'Adj Close' in comp_data.columns:
                    comp_data['Close'] = comp_data['Adj Close']
                elif 'Close_' in [c[:6] for c in comp_data.columns]:
                    # Find the column that starts with 'Close_'
                    close_col = [c for c in comp_data.columns if c.startswith('Close_')][0]
                    comp_data['Close'] = comp_data[close_col]
                    
            comparison_data[crypto] = comp_data
        
        # Normalize data for comparison (starting point = 100)
        fig = go.Figure()
        
        # Add selected crypto
        selected_data = data['Close'] / data['Close'].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=selected_data, 
            mode='lines', 
            name=selected_crypto
        ))
        
        # Add comparison cryptos
        for crypto, crypto_data in comparison_data.items():
            normalized_data = crypto_data['Close'] / crypto_data['Close'].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=crypto_data.index, 
                y=normalized_data, 
                mode='lines', 
                name=crypto
            ))
        
        fig.update_layout(
            title="Normalized Price Comparison (Starting Point = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Disclaimer: This dashboard is for educational purposes only and should not be considered financial advice.")
st.caption("Data source: Yahoo Finance")
