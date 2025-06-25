#project made by Arnav Agrawal, Comments added for explanation
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#Using yfinance for stock data
stock = input("Enter what stock you wanna check. Common stocks: AAPL,MSFT,RELIANCE.NS,TCS.NS \n")  # change this to any stock of your choice
stock_data = yf.download(stock, start='2015-01-01', end='2024-01-01')
input = stock_data[['Open', 'High', 'Low', 'Volume']]
closing = stock_data['Close'].values  # This is what we want to find
# We Normalise input features below for neural networks
scaler = MinMaxScaler()
normalizer = scaler.fit_transform(input)
# Seprating data into testing and training
Xtrain, Xtest, ytrain, ytest = train_test_split(normalizer, closing, test_size=0.2, shuffle=False)
# Building the Neural Network below
model = Sequential([
    Dense(64, activation='relu', input_shape=(Xtrain.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Predicting the closing price
])
model.compile(optimizer='adam', loss='mean_squared_error')
#Training of model Done below
history = model.fit(
    Xtrain, ytrain,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1)
# Prediction maker
ypred = model.predict(Xtest)
error = mean_squared_error(ytest, ypred)
print(f"Test Mean Squared Error: {error:.2f}")
#Currency detection
inr_stock = stock.endswith('.NS') or stock.endswith('.BO')
symbol = '₹'
if not inr_stock:
    try:
        usd_inr = yf.download('USDINR=X', period='1d')['Close'].iloc[-1]
        ypred *= usd_inr
        ytest *= usd_inr
        symbol = '₹ (converted)'
    except Exception as e:
        print("Currency conversion failed Showing USD values.")
        symbol = '$'
#graph of predicted vs actual
plt.figure(figsize=(12, 6))
plt.plot(ytest, label='Actual Closing Price', linewidth=2)
plt.plot(ypred, label='Predicted Closing Price', linestyle='--')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Time Step (Test Set)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#closing price for next day predicted:
latestinput = normalizer[-1].reshape(1, -1)
predictedprice = model.predict(latestinput)[0][0]
print(f"Predicted closing price for next day:{symbol}{predictedprice:.2f}")
