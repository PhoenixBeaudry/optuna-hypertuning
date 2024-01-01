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
from train import upper_shadow, lower_shadow, add_daily_open_feature, add_technical_indicators, scale_data, perform_wavelet_transform, create_dataset, calculate_weighted_rmse, decaying_rmse_loss

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

# Combine all features into a DataFrame
data_df = pd.DataFrame({
    'timestamps': timestamps,
    'close': close_prices,
    'high': high_prices,
    'low': low_prices,
    'volume': volumes
})


# Create more advanced technical indicators
df = add_technical_indicators(data_df)
# remove all NaN values
df.dropna(inplace=True)
df.drop('timestamps', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)

data  = df.values

scaler = MinMaxScaler(feature_range=(0, 1))

num_features = data.shape[1]

# Optuna objective
def objective(trial):
    #Layers
    hidden_units = trial.suggest_int('hidden_units', 128, 1024)
    num_layers = 1 #trial.suggest_int('num_layers', 1, 2)
    layer_multiplier = 1 #trial.suggest_float('layer_multiplier', 0.25, 2.0, step=0.25)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    num_previous_intervals = trial.suggest_int('num_previous_intervals', 30, 170)
    # Elastic Net Regularization hyperparameters
    l1_reg = trial.suggest_float('l1_reg', 1e-5, 1e-2, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

    # Optimizer
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer_type = "ranger"
    total_steps = trial.suggest_int("total_steps", 5000, 20000)
    warmup_proportion = trial.suggest_float("warmup_proportion", 0.05, 0.2)
    min_lr = trial.suggest_float("min_lr", 1e-6, 1e-4, log=True)
    sync_period = trial.suggest_int("sync_period", 5, 10)
    slow_step_size = trial.suggest_float("slow_step_size", 0.4, 0.6)

    # Wavelet
    wavelet_transform = "True"
    wavelet_type = "db4"
    decomposition_level = 4

    #Training
    batch_size = trial.suggest_int('batch_size', 64, 1024, step=128)

    # Create a model with the current trial's hyperparameters
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(hidden_units, return_sequences=num_layers > 1,
                       input_shape=(num_previous_intervals, num_features),
                       kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))  # Apply Elastic Net regularization
        else:
            model.add(LSTM(int(hidden_units*layer_multiplier*i), return_sequences=i < num_layers - 1,
                       kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))  # Apply Elastic Net regularization
        model.add(Dropout(dropout_rate))
    model.add(Dense(100, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))  # Apply Elastic Net 

    if(optimizer_type == "adam"):
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif(optimizer_type == "ranger"):
        radam = tfa.optimizers.RectifiedAdam(
            learning_rate=learning_rate,
            total_steps=total_steps,
            warmup_proportion=warmup_proportion,
            min_lr=min_lr,
        )
        optimizer = tfa.optimizers.Lookahead(radam, sync_period=sync_period, slow_step_size=slow_step_size)
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
study = optuna.create_study(direction='minimize', study_name="regularization", load_if_exists=True, storage=database_url)

# Do the study
study.optimize(objective)  # Adjust the number of trials