import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

def fetch_and_prepare_data(ticker):
    # Set random seed for NumPy and TensorFlow
    np.random.seed(0)
    tf.random.set_seed(0)

    # Fetch historical price data using yfinance for the last 2 years
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(years=2)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    # Drop columns except for 'Date' and 'Close'
    data = data[['Date', 'Close']]
    data.columns = ['Date', 'ltp']

    # Normalize the 'ltp' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    ltp_scaled = scaler.fit_transform(data['ltp'].values.reshape(-1, 1))

    return data, ltp_scaled, scaler

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_and_train_model(X, y, n_steps, n_features):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(X, y, epochs=200, verbose=0)
    
    return model

def make_forecast(model, ltp_scaled, n_steps, scaler):
    # Make predictions for the next 30 days
    forecast = []
    batch = ltp_scaled[-n_steps:].reshape((1, n_steps, 1))
    for i in range(30):
        pred = model.predict(batch)[0]
        forecast.append(pred)
        batch = np.append(batch[:, 1:, :], [[pred]], axis=1)

    # Inverse transform the forecasted prices
    forecast = scaler.inverse_transform(forecast)
    
    return forecast

def get_52_week_high_low(data):
    last_52_weeks = data[-252:]  # Approx 252 trading days in a year
    high_52_week = last_52_weeks['ltp'].max()
    low_52_week = last_52_weeks['ltp'].min()
    return high_52_week, low_52_week

def main(ticker):
    # Fetch and prepare data
    data, ltp_scaled, scaler = fetch_and_prepare_data(ticker)
    
    # Choose the number of time steps
    n_steps = 30

    # Prepare the data
    X, y = prepare_data(ltp_scaled, n_steps)

    # Reshape input data for LSTM [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Build and train the model
    model = build_and_train_model(X, y, n_steps, n_features)

    # Make forecast
    forecast = make_forecast(model, ltp_scaled, n_steps, scaler)

    # Get 52-week high and low
    high_52_week, low_52_week = get_52_week_high_low(data)

    # Generate business dates for the next 30 days
    current_date = pd.Timestamp.now().normalize()
    business_dates = pd.date_range(start=current_date, periods=30, freq='B')

    # Prepare results
    results = {"dates": business_dates, "forecast": forecast.flatten(), "high_52_week": high_52_week, "low_52_week": low_52_week}
    return results

if __name__ == "__main__":
    ticker = input("Enter the ticker symbol: ").strip().upper()
    results = main(ticker)

    # Extract results
    business_dates = results['dates']
    forecast = results['forecast']
    high_52_week = results['high_52_week']
    low_52_week = results['low_52_week']

    # Print the 52-week high and low
    print(f"52-Week High: {high_52_week}")
    print(f"52-Week Low: {low_52_week}")

    # Print the forecasted prices in a table
    print("\nNext 30 days' predictions:")
    print(f"{'Date':<12} {'Forecasted Price':<18}")
    print("-" * 30)
    for date, price in zip(business_dates, forecast):
        print(f"{date.date():<12} {price:<18.2f}")
