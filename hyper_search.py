# Imports
import numpy as np
import pandas as pd
import optuna
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
import time
import tensorflow_addons as tfa
from helper_functions import create_dataset, calculate_weighted_rmse, decaying_rmse_loss


# Optuna objective
def objective(trial):
    print("===== Starting new trial =====")
    #### Data Loading
    
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

    ############# SEARCH PARAMS #############

    #Layers
    hidden_units = trial.suggest_int('hidden_units', 64, 768)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    num_previous_intervals = trial.suggest_int('num_previous_intervals', 30, 170)

    # Elastic Net Regularization hyperparameters
    l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-3, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)

    # Optimizer
    learning_rate = 1e-2
    weight_decay = 1e-4
    sync_period = 7
    slow_step_size = 0.7

    #Training
    batch_size = trial.suggest_int('batch_size', 64, 256, step=64)


    ##########################################

    # Create a model with the current trial's hyperparameters
    model = Sequential()
    model.add(LSTM(hidden_units, return_sequences=False,
                input_shape=(num_previous_intervals, num_features),
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))) # Apply Elastic Net regularization
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))) # Apply Elastic Net 


    # Optimizer
    # AdamW Optimizer
    adamw = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    optimizer = tfa.optimizers.Lookahead(adamw, sync_period=sync_period, slow_step_size=slow_step_size)

    model.compile(optimizer=optimizer, loss=decaying_rmse_loss)

    # Step 1: Split the raw data into training and testing sets
    df_train, df_test = train_test_split(data, test_size=0.05, shuffle=False)

    # Step 2: Initialize and fit the scaler on the training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df_train)

    # Step 3: Scale both the training and testing data using the fitted scaler
    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    # Split the data into X,y sets
    X_train, y_train = create_dataset(df_train_scaled, num_previous_intervals)
    X_test, y_test = create_dataset(df_test_scaled, num_previous_intervals)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    
    print("Beginning model training...")
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stopping, reduce_lr])
    print("Model training finished!")
    
    start = time.time()
    # Evaluate the model
    predictions = model.predict(X_test)
    end = time.time()
    trial.set_user_attr("inference_time", end-start)
    
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

    return rmse


if __name__ == "__main__":

    # Load the .env file
    load_dotenv()

    database_url = os.environ.get('DATABASE_URL')

    #Create Study
    study = optuna.create_study(direction='minimize', study_name="formless-v2-search-4", load_if_exists=True, storage=database_url)

    # Do the study
    study.optimize(objective)  # Adjust the number of trials