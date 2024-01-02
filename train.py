# Imports
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
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

    #return np.array(X), np.array(y)
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


if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    df, data, num_features = get_data('data', '2y_data.pickle')

    ##### Add your hyperparameter options
    #Layers
    hidden_units = 947
    num_layers = 1
    layer_multiplier = 1
    dropout_rate = 0.1352979618569746
    num_previous_intervals = 57

    # Elastic Net Regularization hyperparameters
    l1_reg = 1.2669911888395433e-05
    l2_reg = 1.0003319428243117e-05

    # Optimizer
    learning_rate = 0.002546037429204024
    optimizer_type = "ranger"
    total_steps = 5398
    warmup_proportion = 0.17004444961347331
    min_lr = 1.751520128062939e-06
    sync_period = 6
    slow_step_size = 0.5859028778720838

    # Wavelet
    wavelet_transform = "False"
    wavelet_type = "db4"
    decomposition_level = 4

    #Training
    batch_size = 960

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

    # Create a model with the current trial's hyperparameters
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(hidden_units, return_sequences=num_layers > 1, input_shape=(num_previous_intervals, num_features)))
        else:
            model.add(LSTM(hidden_units, return_sequences=i < num_layers - 1))
        model.add(Dropout(dropout_rate))
    model.add(Dense(100)) # activation='linear'
    model.compile(optimizer=optimizer, loss=decaying_rmse_loss)


    # Assume 'df' is your full dataset

    

    # Step 1: Split the raw data into training and testing sets
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False, random_state=42)

    # Step 2: Apply the wavelet transform to the training and testing data independently
    if wavelet_transform == "True":
        df_train = perform_wavelet_transform(df_train, wavelet=wavelet_type, level=decomposition_level)
        df_test = perform_wavelet_transform(df_test, wavelet=wavelet_type, level=decomposition_level)

    # Step 3: Initialize and fit the scaler on the wavelet-transformed training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = MinMaxScaler()
    scaler = scaler.fit(df_train)

    # Step 4: Scale both the training and testing data using the fitted scaler
    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    # Split the data into X,y sets
    X_train, y_train = create_dataset(df_train_scaled, num_previous_intervals)
    X_test, y_test = create_dataset(df_test_scaled, num_previous_intervals)



    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1, verbose=0, callbacks=[early_stopping])

    save_scaler_as_pickle(scaler)

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

    print(f"Models RMSE: {rmse}")

    print(original_scale_predictions)

    # Save the trained model
    model.save('trained_models/hyper_model.h5')