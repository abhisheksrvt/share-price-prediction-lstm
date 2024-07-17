import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import ta  # Technical Analysis library
from sklearn.model_selection import train_test_split
import os

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data, ticker, look_back=60):
    if data.empty:
        raise ValueError("No data downloaded. Please check your data source.")
    
    # Feature Engineering
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd_diff()
    
    # Filling NaN values
    data.fillna(0, inplace=True)
    
    features = ['Close', 'MA50', 'MA200', 'RSI', 'MACD']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predicting the 'Close' price
    
    X, y = np.array(X), np.array(y)
    
    # Save scaler for future use
    with open(f'scaler_{ticker}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    start_date = pd.Timestamp(data.index[0])
    end_date = pd.Timestamp(data.index[-1])
    
    return X, y, scaler, start_date, end_date

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, ticker):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((X.shape[1], X.shape[2]))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # Save the model in the new .keras format with the stock name in the filename
    model.save(f'stock_model_{ticker}.keras')
    
    return model, history

def predict(model, X, future_days=30):
    # Predict using the last sequence in X
    last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
    future_predictions = []
    
    for _ in range(future_days):
        prediction = model.predict(last_sequence)[0][0]
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = prediction
    
    return future_predictions

def plot_results(data, y_pred, future_pred, dates, history=None):
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    if history:
        plt.subplot(3, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.grid(True)

    # Plot true prices
    plt.subplot(3, 1, 2)
    plt.plot(dates, data['Close'], label='True Prices', linestyle='-', linewidth=1.5, color='blue')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.grid(True)
    
    # Plot predicted prices
    predicted_start = len(dates) - len(y_pred)
    plt.plot(dates[predicted_start:], y_pred, label='Predicted Prices', linestyle='-', linewidth=1.0, color='orange')
    plt.legend()
    plt.grid(True)
    
    # Plot future predicted prices
    plt.subplot(3, 1, 3)
    future_dates = pd.date_range(start=dates[-1], periods=len(future_pred)+1, freq='B')[1:]
    plt.plot(future_dates, future_pred, label='Future Predictions', linestyle='--', linewidth=1.0, color='green')
    plt.legend(loc='upper left')  # Set legend to upper left
    plt.xlabel('Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_future_predictions(future_predictions, end_date):
    last_date = pd.Timestamp(end_date)
    print("\nPredicted Prices for the next 30 days:")
    for i, prediction in enumerate(future_predictions, start=1):
        next_date = last_date + pd.DateOffset(days=i)
        print(f"{next_date.date()}: {prediction:.2f}")

def main():
    ticker = 'RELIANCE.NS'  # Change to the desired NSE stock symbol
    start_date = '2010-01-01'  # Default start date (can be overridden)
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')  # Automatically set end date to today
    look_back = 60
    
    # Download data
    data = download_data(ticker, start_date, end_date)
    
    # Preprocess data and retrieve start_date and end_date from data
    X, y, scaler, start_date, end_date = preprocess_data(data, ticker, look_back)
    
    # Check if data is available for training
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data available for training.")
    
    # Check if the model file exists
    model_filename = f'stock_model_{ticker}.keras'
    if os.path.exists(model_filename):
        model = load_model(model_filename)
        print(f"Loaded existing model from {model_filename}.")
    else:
        # Train model if no existing model found
        model, history = train_model(X, y, ticker)
    
    # Predict future prices
    future_predictions = predict(model, X, future_days=30)
    
    # Inverse transform all y values for printing
    y_pred_scaled = model.predict(X).reshape(-1, 1)
    y_pred_padded = np.hstack((y_pred_scaled, np.zeros((len(y_pred_scaled), 4))))
    y_pred = scaler.inverse_transform(y_pred_padded)[:, 0]

    future_pred_scaled = np.array(future_predictions).reshape(-1, 1)
    future_pred_padded = np.hstack((future_pred_scaled, np.zeros((len(future_pred_scaled), 4))))
    future_pred = scaler.inverse_transform(future_pred_padded)[:, 0]
    
    # Print future predictions along with dates
    print_future_predictions(future_pred, end_date)
    
    # Plotting results with dates on x-axis
    plot_results(data, y_pred, future_pred, data.index, history if 'history' in locals() else None)

if __name__ == "__main__":
    main()
