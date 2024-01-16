# Imports
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

def get_data_struct_from_pickle(filepath):
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



def get_data(filepath = 'data/1y_data.pickle'):
    data_structure = get_data_struct_from_pickle(filepath)

    data_df = data_struct_to_data_frame(data_structure)
    
    df = add_technical_indicators(data_df)
    
    df = clean_data(df)

    return df


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
    df['hour'] = df['timestamps'].dt.hour
    df['day_of_week'] = df['timestamps'].dt.dayofweek

    return df


# Function to create dataset
def create_dataset(data, input_time_steps=100, future_interval_to_predict=100):
    data = np.array(data)
    X, y = [], []
    for i in range(len(data) - input_time_steps - future_interval_to_predict):
        X.append(data[i:(i + input_time_steps), :])
        y.append(data[(i + input_time_steps + future_interval_to_predict - 1), 0])
    return np.array(X), np.array(y)

# Function to create test dataset
def create_test_dataset(data, input_time_steps=100, future_interval_to_predict=100):
    data = np.array(data)
    X, y = [], []
    for i in range(len(data) - input_time_steps - future_interval_to_predict):
        X.append(data[i:(i + input_time_steps), :])
        y.append(data[(i + input_time_steps):(i + input_time_steps + future_interval_to_predict), 0])
    return np.array(X), np.array(y)



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
    k = 0.001
    # Create weights array
    weights = tf.exp(-k * tf.range(tf.shape(y_pred)[1], dtype=tf.float32))
    # Calculate weighted squared errors for each row
    weighted_squared_errors = tf.square(y_pred - y_true) * weights
    # Sum the weighted squared errors and the weights for each row
    sum_weighted_squared_errors = tf.reduce_sum(weighted_squared_errors, axis=1)
    sum_weights = tf.reduce_sum(weights)
    # Calculate RMSE for each row
    rmse_per_row = tf.sqrt(sum_weighted_squared_errors / sum_weights)
    # Calculate the mean of the RMSE values for each row
    mean_rmse = tf.reduce_mean(rmse_per_row)
    return mean_rmse

