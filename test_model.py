# Imports
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pywt
from dotenv import load_dotenv
import os
import tensorflow_addons as tfa


# Load our model
model = tf.keras.models.load_model(f'trained_models/hyper_model.h5', compile=False)
# Load our scaler
with open(f'trained_models/hyper_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Load the data from the pickle file
with open('data/2y_data.pickle', 'rb') as file:
    data_structure = pickle.load(file)

data_structure = [data_structure[0],
data_structure[1],
data_structure[2],
data_structure[3],
data_structure[4]]

data_array = np.array(data_structure).T


#Assuming the structure is [timestamps, close, high, low, volume]
timestamps = pd.to_datetime(data_array[:, 0], unit='ms')
close_prices = data_array[:, 1]
high_prices = data_array[:, 2]
low_prices = data_array[:, 3]
volumes = data_array[:, 4]

# Combine all features into a DataFrame
data_df = pd.DataFrame({
    'timestamps': timestamps,
    'close': close_prices,
    'high': high_prices,
    'low': low_prices,
    'volume': volumes
})

# Technical indicator helper functions
def upper_shadow(df): return df['high'] - np.maximum(df['close'], df['open'])
def lower_shadow(df): return np.minimum(df['close'], df['open']) - df['low']
def add_daily_open_feature(df):
    # Convert timestamps to dates if necessary (assuming 'timestamps' is in datetime format)
    df['date'] = df['timestamps'].dt.date
    # Group by date and take the last 'close' for each day
    daily_open_prices = df.groupby('date')['close'].last()
    # Shift the daily_open_prices to use the last 'close' as the 'open' for the following day.
    daily_open_prices = daily_open_prices.shift(1)
    # Map the daily_open_prices to each corresponding timestamps
    df['open'] = df['date'].map(daily_open_prices)
    # Handle the 'open' for the first day in the dataset
    df['open'].bfill(inplace=True)
    return df

# Create more advanced technical indicators
def add_technical_indicators(df):
    # Simple Moving Average
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_15'] = df['close'].rolling(window=15).mean()

    # Exponential Moving Average
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_15'] = df['close'].ewm(span=15, adjust=False).mean()

    # Relative Strength Index
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))

    # Moving Average Convergence Divergence
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Make sure the 'timestamps' column is a datetime type before applying the function
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = add_daily_open_feature(df)
    df['upper_Shadow'] = upper_shadow(df)
    df['lower_Shadow'] = lower_shadow(df)
    df["high_div_low"] = df["high"] / df["low"]
    df['trade'] = df['close'] - df['open']
    df['shadow1'] = df['trade'] / df['volume']
    df['shadow3'] = df['upper_Shadow'] / df['volume']
    df['shadow5'] = df['lower_Shadow'] / df['volume']
    df['mean1'] = (df['shadow5'] + df['shadow3']) / 2
    df['mean2'] = (df['shadow1'] + df['volume']) / 2

    return df

df = add_technical_indicators(data_df)
# remove all NaN values
df.dropna(inplace=True)
df.drop('timestamps', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)

data  = df.values

num_features = data.shape[1]

# Function to scale dataset
def scale_data(data):
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def perform_wavelet_transform(data, wavelet='db4', level=4):
    # Initialize an empty list to store the denoised features
    data_denoised_list = []

    # Apply wavelet transform to each feature separately
    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:, i], wavelet=wavelet, level=level)
        # Zero out the high-frequency components for denoising
        coeffs[1:] = [np.zeros_like(coeff) for coeff in coeffs[1:]]
        # Reconstruct the denoised signal
        data_denoised = pywt.waverec(coeffs, wavelet)
        # Append the denoised feature to the list
        data_denoised_list.append(data_denoised)

    # Combine the denoised features back into a single array
    data_denoised_combined = np.column_stack(data_denoised_list)
    return data_denoised_combined

wavelet_type = "db4"
decomposition_level = 4
input = scale_data(perform_wavelet_transform(data, wavelet=wavelet_type, level=decomposition_level))

# Evaluate the model
predictions = model.predict(input[-57:].reshape([1, 57, num_features]))

# This is literally fucking stupid. How does ML work like this.
# Create a zero-filled array with the same number of samples and timesteps
modified_predictions = np.zeros((predictions.shape[0], predictions.shape[1], num_features))
# Place predictions into the first feature of this array
modified_predictions[:, :, 0] = predictions
modified_predictions_reshaped = modified_predictions.reshape(-1, num_features)
# Apply inverse_transform
original_scale_predictions = scaler.inverse_transform(modified_predictions_reshaped)
# Reshape back to original predictions shape, if needed
original_scale_predictions = original_scale_predictions[:, 0].reshape(predictions.shape[0], predictions.shape[1])

predicted_closes = original_scale_predictions[0].tolist()

print(predicted_closes)