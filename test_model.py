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
from train import scale_data, perform_wavelet_transform, get_data, load_scaler_from_pickle


if __name__ == "__main__":
    # Load our model
    model = tf.keras.models.load_model(f'trained_models/hyper_model.h5', compile=False)
    # Load our scaler
    scaler = load_scaler_from_pickle()

    df, data, num_features = get_data('data', '2y_data.pickle')

    wavelet_type = "db4"
    decomposition_level = 4
    input = scale_data(perform_wavelet_transform(data, wavelet=wavelet_type, level=decomposition_level), scaler)

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