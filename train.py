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
from tensorflow.keras.regularizers import l1_l2
from dotenv import load_dotenv
import tensorflow_addons as tfa
from helper_functions import create_dataset, calculate_weighted_rmse, decaying_rmse_loss, get_data



if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    df, data, num_features = get_data('data', '2y_data.pickle')

    ##### Add your hyperparameter options
    #Layers
    hidden_units = 512
    num_layers = 1
    layer_multiplier = 1
    dropout_rate = 0.2
    num_previous_intervals = 100

    # Elastic Net Regularization hyperparameters
    l1_reg = 3.624635038622804e-05
    l2_reg = 0.0012312129801873588

    # Optimizer
    learning_rate = 0.005551452391723501
    total_steps = 10000
    warmup_proportion = 0.10323494107850861
    min_lr = 7.291141404840904e-05
    sync_period = 5
    slow_step_size = 0.4474053367229199


    #Training
    batch_size = 64

    # Optimizer
    radam = tfa.optimizers.RectifiedAdam(
        learning_rate=learning_rate,
        total_steps=total_steps,
        warmup_proportion=warmup_proportion,
        min_lr=min_lr,
    )
    optimizer = tfa.optimizers.Lookahead(radam, sync_period=sync_period, slow_step_size=slow_step_size)


    # Create a model with the current trial's hyperparameters
    model = Sequential()

    model.add(LSTM(hidden_units, return_sequences=num_layers > 1, input_shape=(num_previous_intervals, num_features), kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))) # Apply Elastic Net regularization
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))) # Apply Elastic Net 
    model.compile(optimizer=optimizer, loss=decaying_rmse_loss)


    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False, random_state=42)

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stopping])

    # Save the scaler
    with open('trained_models/formless-v2_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Evaluate the model
    predictions = model.predict(X_test)
    # This is literally fucking stupid. How does ML work like this.
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
    
    # Save the trained model
    model.save('trained_models/formless-v2.h5')