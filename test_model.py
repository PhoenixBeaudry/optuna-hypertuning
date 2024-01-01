# Imports
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import pywt
from dotenv import load_dotenv
import os
import tensorflow_addons as tfa
from train import upper_shadow, lower_shadow, add_daily_open_feature, add_technical_indicators, scale_data, perform_wavelet_transform, create_dataset, calculate_weighted_rmse, decaying_rmse_loss

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



df = add_technical_indicators(data_df)
# remove all NaN values
df.dropna(inplace=True)
df.drop('timestamps', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)

data  = df.values

num_features = data.shape[1]


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