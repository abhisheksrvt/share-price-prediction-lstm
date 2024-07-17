# Stock Price Forecasting with LSTM
This repository contains a script for forecasting stock prices using Long Short-Term Memory (LSTM) networks. 
The project leverages historical stock price data and technical indicators to predict future stock prices. 
This project is for educational and research purposes only.

# Disclaimer
This project is intended for educational and research purposes only. It should not be used for making any investment decisions. 
The predictions provided by this model are not financial advice. Do not use this model to buy or sell stocks.

# Features
- Data Download: Fetch historical stock price data using yfinance.
- Data Preprocessing: Feature engineering with technical indicators (MA50, MA200, RSI, MACD) and scaling.
- Model Building: LSTM-based neural network model for stock price prediction.
- Model Training: Training the LSTM model with historical data.
- Prediction: Predict future stock prices for the next 30 days.
- Visualization: Plot true prices, predicted prices, and future predictions.

# Results 

![reliance_lstm](https://github.com/user-attachments/assets/8bc65c0e-e17f-4675-96fb-298a4084b899)


# Installation
To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/abhisheksrvt/share-price-prediction-lstm.git
cd share-price-prediction-lstm
pip install -r requirements.txt
```
# Algorithm Explanation
# 1. Download Data
The script starts by downloading historical stock price data using the yfinance library.
```python
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data
```

# 2. Preprocess Data
The downloaded data is then preprocessed to include technical indicators such as Moving Averages (MA50, MA200), 
Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD). 
The data is scaled using MinMaxScaler to normalize the feature values.

```python
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
```

# 3. Build Model
The LSTM model is built using the Sequential model from Keras. The model consists of LSTM layers and Dropout layers to prevent overfitting, 
followed by a Dense layer to produce the output.

```python
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```

# 4. Train Model
The data is split into training and validation sets. The model is then trained using the training set, and the validation set 
is used to monitor the model's performance during training.

```python
def train_model(X, y, ticker):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((X.shape[1], X.shape[2]))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # Save the model in the new .keras format with the stock name in the filename
    model.save(f'stock_model_{ticker}.keras')
    
    return model, history
```

# 5. Predict Future Prices
The trained model is used to predict future stock prices. The last sequence in the input data is used to generate future predictions iteratively.

```python
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
```

# 6. Plot Results
The results are visualized by plotting the true stock prices, predicted prices, and future predictions.

```python
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
```

# 7. Print Future Predictions
The future predicted prices are printed along with their respective dates.

```python
def print_future_predictions(future_predictions, end_date):
    last_date = pd.Timestamp(end_date)
    print("\nPredicted Prices for the next 30 days:")
    for i, prediction in enumerate(future_predictions, start=1):
        next_date = last_date + pd.DateOffset(days=i)
        print(f"{next_date.date()}: {prediction:.2f}")
```

# Example
Here's a basic example of how to use the script:

```python
def main():
    ticker = 'RELIANCE.NS'  # Change to the desired NSE stock symbol
    start_date = '2010-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    look_back = 60

    # Download data
    data = download_data(ticker, start_date, end_date)

    # Preprocess data
    X, y, scaler, start_date, end_date = preprocess_data(data, ticker, look_back)

    # Train model
    model, history = train_model(X, y, ticker)

    # Predict future prices
    future_predictions = predict(model, X, future_days=30)

    # Plot results
    plot_results(data, y_pred, future_pred, data.index, history)
    
    # Print future predictions
    print_future_predictions(future_predictions, end_date)

if __name__ == "__main__":
    main()
```

# Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.

# License
This project is licensed under the MIT License. See the LICENSE file for details.











