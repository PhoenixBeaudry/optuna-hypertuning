# Imports
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1_l2
from dotenv import load_dotenv
import tensorflow_addons as tfa
from helper_functions import create_dataset, calculate_weighted_rmse, decaying_rmse_loss, clean_data



if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    #Load the data from the pickle file
    with open("data/1y_data.pickle", 'rb') as file:
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

    # remove all NaN values
    data_df.dropna(inplace=True)
    data_df.drop('timestamps', axis=1, inplace=True)

    data = data_df.values
    num_features = data.shape[1]

    ##### Add your hyperparameter options
    #Layers
    hidden_units = 800
    dropout_rate = 0.2
    num_previous_intervals = 75

    # Elastic Net Regularization hyperparameters
    l1_reg = 1e-5
    l2_reg = 1e-3

    # Optimizer
    learning_rate = 1e-3
    weight_decay = 1e-4
    sync_period = 5
    slow_step_size = 0.4


    #Training
    batch_size = 128

    # Optimizer
    # AdamW Optimizer
    adamw = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    optimizer = tfa.optimizers.Lookahead(adamw, sync_period=sync_period, slow_step_size=slow_step_size)

    # Create a model with the current trial's hyperparameters
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=False, input_shape=(num_previous_intervals, num_features), kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))) # Apply Elastic Net regularization
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='linear', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))) # Apply Elastic Net 
    model.compile(optimizer=optimizer, loss=decaying_rmse_loss)


    df_train, df_test = train_test_split(data, test_size=0.2, shuffle=False)

    # Step 3: Initialize and fit the scaler on the wavelet-transformed training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df_train)

    # Step 4: Scale both the training and testing data using the fitted scaler
    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    # Split the data into X,y sets
    X_train, y_train = create_dataset(df_train_scaled, num_previous_intervals)
    X_test, y_test = create_dataset(df_test_scaled, num_previous_intervals)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('trained_models/formless-v2_1.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stopping, model_checkpoint, reduce_lr])

    # Save the scaler
    with open('trained_models/formless-v2_1_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Evaluate the model
    predictions = model.predict(X_test)
    # Create a zero-filled array with the same number of samples and timesteps
    modified_predictions = np.zeros((predictions.shape[0], predictions.shape[1], num_features))
    # Place predictions into the first feature of this array
    modified_predictions[:, :, 0] = predictions
    # Reshape modified_predictions for inverse_transform
    modified_predictions_reshaped = modified_predictions.reshape(-1, num_features)
    # Apply inverse_transform
    original_scale_predictions = scaler.inverse_transform(modified_predictions_reshaped)
    # Reshape back to original predictions shape, if needed
    original_scale_predictions = original_scale_predictions[:, 0].reshape(predictions.shape[0], predictions.shape[1])
    # Create a zero-filled array with the same number of samples and timesteps
    modified_y_test = np.zeros((y_test.shape[0], y_test.shape[1], num_features))
    # Place y_test into the first feature of this array
    modified_y_test[:, :, 0] = y_test
    # Reshape modified_y_test for inverse_transform
    modified_y_test_reshaped = modified_y_test.reshape(-1, num_features)
    # Apply inverse_transform
    original_scale_y_test = scaler.inverse_transform(modified_y_test_reshaped)
    # Reshape back to original y_test shape, if needed
    # (Selecting only the first feature, assuming y_test corresponds to the first feature)
    original_scale_y_test = original_scale_y_test[:, 0].reshape(y_test.shape[0], y_test.shape[1])
    rmse = calculate_weighted_rmse(original_scale_predictions, original_scale_y_test)

    print(f"Models RMSE: {rmse}")
    