# Imports
import numpy as np
import pandas as pd
import optuna
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

def get_data_struct_from_pickle(directory, filename):
    filepath = os.path.join(directory, filename)

    #Load the data from the pickle file
    with open(filepath, 'rb') as file:
        data_structure = pickle.load(file)

    data_structure = [data_structure[0],
    data_structure[1],
    data_structure[2],
    data_structure[3],
    data_structure[4]]

    return data_structure


def data_struct_to_data_frame(data_structure):
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

    return data_df


def clean_data(df):
    # remove all NaN values
    df.dropna(inplace=True)
    df.drop('timestamps', axis=1, inplace=True)
    df.drop('date', axis=1, inplace=True)

    return df


"""
Take a raw data_structure, transform it to a data frame, add features, and clean it
"""
def data_pipeline(data_structure):
    data_df = data_struct_to_data_frame(data_structure)
    
    df = add_technical_indicators(data_df)
    
    df = clean_data(df)

    data = df.values    

    num_features = data.shape[1]

    return df, data, num_features


def get_data(directory = 'data', filename = '2y_data.pickle'):
    data_structure = get_data_struct_from_pickle(directory, filename)

    df, data, num_features = data_pipeline(data_structure)

    return df, data, num_features


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


# Function to scale dataset
def scale_data(data, scaler):
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def perform_wavelet_transform(data, wavelet='db1', level=2):
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


# Function to create dataset
def create_dataset(df, input_time_steps=100, future_intervals=100):

    # This is the equivelant to making a sliding window. For every X, the y will have the next 100 closes.
    total_size = len(df) - input_time_steps - future_intervals + 1
    X = np.lib.stride_tricks.as_strided(
        df,
        shape=(total_size, input_time_steps, df.shape[1]),
        strides=(df.strides[0], df.strides[0], df.strides[1])
    )
    y = np.lib.stride_tricks.as_strided(
        df[:, 0],  # Assuming 'close' is the first feature
        shape=(total_size, future_intervals),
        strides=(df.strides[0], df.strides[0])
    )

    return X, y


# Scoring function for model
def calculate_weighted_rmse(predictions: np, actual: np) -> float:
    predictions = np.array(predictions)
    actual = np.array(actual)

    k = 0.001

    # Create weights array
    weights = np.exp(-k * np.arange(predictions.shape[1]))
    
    # Calculate weighted squared errors for each row
    weighted_squared_errors = (predictions - actual) ** 2 * weights
    
    # Sum the weighted squared errors and the weights for each row
    sum_weighted_squared_errors = np.sum(weighted_squared_errors, axis=1)
    sum_weights = np.sum(weights)
    
    # Calculate RMSE for each row
    rmse_per_row = np.sqrt(sum_weighted_squared_errors / sum_weights)

    # Calculate the mean of the RMSE values for each row
    mean_rmse = np.mean(rmse_per_row)
    
    return mean_rmse


# Objective function to be optimized
def decaying_rmse_loss(y_true, y_pred):
    k = 0.001  # decay rate, adjust as needed

    # Ensure predictions and actual values are tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Calculate the weights with exponential decay
    seq_length = tf.shape(y_true)[1]  # assuming y_true is of shape [batch_size, sequence_length]
    weights = tf.exp(-k * tf.range(start=0.0, limit=tf.cast(seq_length, tf.float32), dtype=tf.float32))

    # Compute weighted squared error
    squared_errors = tf.square(y_pred - y_true)
    weighted_squared_errors = weights * squared_errors

    # Compute the weighted RMSE
    weighted_rmse = tf.sqrt(tf.reduce_sum(weighted_squared_errors) / tf.reduce_sum(weights))

    return weighted_rmse


def save_scaler_as_pickle(scaler):
    directory = 'trained_models'
    filename = 'hyper_scaler.pkl'
    filepath = os.path.join(directory, filename)

    # Make sure the directory exists
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Save the scaler
    with open(filepath, 'wb') as file:
        pickle.dump(scaler, file)


def load_scaler_from_pickle(directory = 'trained_models', filename = 'hyper_scaler.pkl'):
    filepath = os.path.join(directory, filename)

    with open(filepath, 'rb') as file:
        scaler = pickle.load(file)
    
    return scaler
