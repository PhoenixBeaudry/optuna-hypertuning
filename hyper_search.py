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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pywt
from dotenv import load_dotenv
import os
import tensorflow_addons as tfa

# Load the .env file
load_dotenv()

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
"""
#Combine all features into a single array
data_combined = np.column_stack((timestamps, close_prices, high_prices, low_prices, volumes))

# Assuming data_array is your numpy array with columns in the order: timestamps, close, high, low, volume
column_names = ['close', 'high', 'low', 'volume']
data_df = pd.DataFrame(data_combined, columns=column_names)
"""
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
    df['open'].fillna(method='bfill', inplace=True)

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
    
    df_feat = df[['timestamps', 'high', 'low', 'close', 'volume']].copy()
    # Make sure the 'timestamps' column is a datetime type before applying the function
    df_feat['timestamps'] = pd.to_datetime(df_feat['timestamps'])
    df_feat = add_daily_open_feature(df_feat)
    
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["high"] / df_feat["low"]
    df_feat['trade'] = df_feat['close'] - df_feat['open']
    #df_feat['gtrade'] = df_feat['trade'] / df_feat['Count']
    df_feat['shadow1'] = df_feat['trade'] / df_feat['volume']
    df_feat['shadow3'] = df_feat['upper_Shadow'] / df_feat['volume']
    df_feat['shadow5'] = df_feat['lower_Shadow'] / df_feat['volume']
    #df_feat['diff1'] = df_feat['volume'] - df_feat['Count']
    df_feat['mean1'] = (df_feat['shadow5'] + df_feat['shadow3']) / 2
    df_feat['mean2'] = (df_feat['shadow1'] + df_feat['volume']) / 2
    #df_feat['mean3'] = (df_feat['trade'] + df_feat['gtrade']) / 2
    #df_feat['mean4'] = (df_feat['diff1'] + df_feat['upper_Shadow']) / 2
    #df_feat['mean5'] = (df_feat['diff1'] + df_feat['lower_Shadow']) / 2
    
    return df

df = add_technical_indicators(data_df)
# remove all NaN values
df.dropna(inplace=True)

final_data_df = df[['close', 'high', 'low', 'volume', 'SMA_5', 'SMA_15',
       'EMA_5', 'EMA_15', 'RSI', 'MACD', 'Signal_Line']]
data  = final_data_df.values

scaler = MinMaxScaler(feature_range=(0, 1))

# Function to scale dataset
def scale_data(data):
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
def create_dataset(data, input_time_steps=100, future_intervals=100):
    X, y = [], []
    for i in range(len(data) - input_time_steps - future_intervals):
        X.append(data[i:(i + input_time_steps), :])
        y.append(data[(i + input_time_steps):(i + input_time_steps + future_intervals), 0])
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


num_features = data.shape[1]

# Optuna objective
def objective(trial):
    hidden_units = trial.suggest_int('hidden_units', 128, 1024)
    num_layers = 1
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 128, 1024, step=128)
    num_previous_intervals = trial.suggest_int('num_previous_intervals', 50, 100)
    wavelet_transform = True
    wavelet_type = trial.suggest_categorical("wavelet_type", ["db1", "db4"])
    decomposition_level = 4
    optimizer_type = trial.suggest_categorical("optimizer", ["adam", "ranger"])

    # Create a model with the current trial's hyperparameters
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(hidden_units, return_sequences=num_layers > 1, input_shape=(num_previous_intervals, num_features)))
        else:
            model.add(LSTM(hidden_units, return_sequences=i < num_layers - 1))
        model.add(Dropout(dropout_rate))
    model.add(Dense(100))

    if(optimizer_type == "adam"):
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif(optimizer_type == "ranger"):
        radam = tfa.optimizers.RectifiedAdam(
            learning_rate=learning_rate,
            total_steps=10000,
            warmup_proportion=0.1,
            min_lr=1e-5,
        )
        optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    else:
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=decaying_rmse_loss)
    
    # Data prep
    if wavelet_transform == "True": 
        X, y = create_dataset(scale_data(perform_wavelet_transform(data, wavelet=wavelet_type, level=decomposition_level)), num_previous_intervals, 100)
    else: 
        X, y = create_dataset(scale_data(data), num_previous_intervals, 100)
    
    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1, verbose=0, callbacks=[early_stopping])

    # Evaluate the model
    predictions = model.predict(X_test)
    # This is literally fucking stupid. How does ML work like this.
    # Create a zero-filled array with the same number of samples and timesteps, but with 5 features
    modified_predictions = np.zeros((predictions.shape[0], predictions.shape[1], num_features))
    # Place predictions into the first feature of this array
    modified_predictions[:, :, 0] = predictions
    # Reshape modified_predictions to 2D (51911*100, 5) for inverse_transform
    modified_predictions_reshaped = modified_predictions.reshape(-1, num_features)
    # Apply inverse_transform
    original_scale_predictions = scaler.inverse_transform(modified_predictions_reshaped)
    # Reshape back to original predictions shape, if needed
    original_scale_predictions = original_scale_predictions[:, 0].reshape(predictions.shape[0], predictions.shape[1])
    # Create a zero-filled array with the same number of samples and timesteps, but with 5 features
    modified_y_test = np.zeros((y_test.shape[0], y_test.shape[1], num_features))
    # Place y_test into the first feature of this array
    modified_y_test[:, :, 0] = y_test
    # Reshape modified_y_test to 2D (51911*100, 5) for inverse_transform
    modified_y_test_reshaped = modified_y_test.reshape(-1, num_features)
    # Apply inverse_transform
    original_scale_y_test = scaler.inverse_transform(modified_y_test_reshaped)
    # Reshape back to original y_test shape, if needed
    # (Selecting only the first feature, assuming y_test corresponds to the first feature)
    original_scale_y_test = original_scale_y_test[:, 0].reshape(y_test.shape[0], y_test.shape[1])
    rmse = calculate_weighted_rmse(original_scale_predictions, original_scale_y_test)

    return rmse


database_url = os.environ.get('DATABASE_URL')

#Create Study
study = optuna.create_study(direction='minimize', study_name="hyper-search-optimizer", load_if_exists=True, storage=database_url)

# Do the study
study.optimize(objective, n_trials=50)  # Adjust the number of trials

# Get the best hyperparameters
best_params = study.best_params
print("Best parameters:", best_params)