# Imports
import numpy as np
import pandas as pd
import optuna
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
import time
import tensorflow_addons as tfa
from helper_functions import create_dataset, calculate_weighted_rmse, decaying_rmse_loss, get_data, create_test_dataset


# Optuna objective
def objective(trial):
    print("===== Starting new trial =====")
    #### Data Loading
    df = get_data('data/1y_data.pickle')
    training_data = df[['close','high','low','volume','EMA_10','EMA_20','RSI','MACD','Signal_Line','mean1','mean2','hour','day_of_week']]
    data = training_data.values
    num_features = data.shape[1]

    ############# SEARCH PARAMS #############
    #Layers
    bidirectional = False
    num_layers = 1
    layer_multiplier = 1
    hidden_units = trial.suggest_int('hidden_units', 300, 700)
    dropout_rate = trial.suggest_float('dropout_rate', 0.18, 0.23)
    num_previous_intervals = trial.suggest_int('num_previous_intervals', 20, 150)
    
    
    # Elastic Net Regularization hyperparameters
    elastic_net = False #trial.suggest_categorical('elastic_net', [True, False])
    l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-1)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-1)


    # Optimizer
    optimizer_type = 'ranger' #trial.suggest_categorical('optimizer_type', ['adam', 'ranger'])
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.03)
    sync_period = trial.suggest_int('sync_period', 8, 9)
    slow_step_size = trial.suggest_float('slow_step_size', 0.4, 0.9)
    total_steps = trial.suggest_int('total_steps', 11000, 20000)
    warmup_proportion = trial.suggest_float('warmup_proportion', 0.1, 0.9)
    min_lr = trial.suggest_float('min_lr', 1e-8, 1e-6)

    if optimizer_type == 'ranger':
        radam = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, total_steps=total_steps, warmup_proportion=warmup_proportion, min_lr=min_lr)
        optimizer = tfa.optimizers.Lookahead(radam, sync_period=sync_period, slow_step_size=slow_step_size)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    #Callbacks
    lr_reduction_factor = trial.suggest_float('lr_reduction_factor', 0.25, 0.35)

    #Training
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 768])

    ##########################################
    print(f"Trial has these parameters: {trial.params}")

    ##### Training

    # Create a model with the current trial's hyperparameters
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            if elastic_net:
                if bidirectional:
                    model.add(Bidirectional(LSTM(hidden_units, return_sequences=num_layers > 1,
                            input_shape=(num_previous_intervals, num_features),
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))) # Apply Elastic Net regularization
                else:
                    model.add((LSTM(hidden_units, return_sequences=num_layers > 1,
                            input_shape=(num_previous_intervals, num_features),
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))) # Apply Elastic Net regularization
            else:
                if bidirectional:
                    model.add(Bidirectional(LSTM(hidden_units, return_sequences=num_layers > 1,
                            input_shape=(num_previous_intervals, num_features))))
                else:
                    model.add((LSTM(hidden_units, return_sequences=num_layers > 1,
                            input_shape=(num_previous_intervals, num_features))))
        else:
            if elastic_net:
                if bidirectional:
                    model.add(Bidirectional(LSTM(int(hidden_units*layer_multiplier*i), return_sequences=i < num_layers - 1,
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))) # Apply Elastic Net regularization
                else:
                    model.add((LSTM(int(hidden_units*layer_multiplier*i), return_sequences=i < num_layers - 1,
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))) # Apply Elastic Net regularization
            else:
                if bidirectional:
                    model.add(Bidirectional(LSTM(int(hidden_units*layer_multiplier*i), return_sequences=i < num_layers - 1)))
                else:
                    model.add(LSTM(int(hidden_units*layer_multiplier*i), return_sequences=i < num_layers - 1))
        model.add(Dropout(dropout_rate))
    if elastic_net:
        model.add(Dense(1, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))) # Apply Elastic Net
    else:
        model.add(Dense(1))
    
    model.compile(optimizer=optimizer, loss="mse")

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
    X_test, y_test = create_test_dataset(df_test_scaled, num_previous_intervals)
    X_test_inverse, y_test_inverse = create_test_dataset(df_test, num_previous_intervals)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_reduction_factor, patience=4, verbose=1)
    #pruning = TFKerasPruningCallback(trial, monitor='val_loss')
    
    print("Beginning model training...")
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])
    print("Model training finished!")
    

    ##### Prediction and Evaluation
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
    # Reshape back to original predictions shape
    original_scale_predictions = original_scale_predictions[:, 0].reshape(predictions.shape[0], predictions.shape[1]).flatten().tolist()
    
    pred_list = []
    for i in range(X_test.shape[0]):
        # Interp between last seen value and prediction
        full_pred = np.linspace(X_test_inverse[i][-1][0], original_scale_predictions[i], 100)
        pred_list.append(full_pred)

    rmse = calculate_weighted_rmse(pred_list, y_test_inverse)

    return rmse


if __name__ == "__main__":
    local = False

    # Load the .env file
    load_dotenv()

    database_url = os.environ.get('DATABASE_URL')

    if local:
        study = optuna.create_study(direction='minimize')
    else:
        #Create Study
        study = optuna.create_study(direction='minimize', study_name="formless-v3-single-interval-new-ema", load_if_exists=True, storage=database_url)

    # Do the study
    study.optimize(objective)