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
from helper_functions import scale_data, perform_wavelet_transform, create_dataset, calculate_weighted_rmse, decaying_rmse_loss, get_data, save_scaler_as_pickle



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