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
from helper_functions import create_dataset, calculate_weighted_rmse, get_data, create_test_dataset



if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    #### Data Loading
    df = get_data()
    training_data = df[['close','high','low','volume','EMA_10','EMA_20','RSI','MACD','Signal_Line','mean1','mean2','hour','day_of_week']]
    data = training_data.values
    num_features = data.shape[1]


    ############# SEARCH PARAMS #############
    #Layers
    bidirectional = False
    num_layers = 1 
    layer_multiplier = 1 
    hidden_units = 100
    dropout_rate = 0.16300714101033043
    num_previous_intervals = 50
    
    # Elastic Net Regularization hyperparameters
    elastic_net = False
    l1_reg = 0 
    l2_reg = 0

    # Optimizer
    optimizer_type = 'ranger' #trial.suggest_categorical('optimizer_type', ['adam', 'ranger'])
    learning_rate = 0.01312969026939553
    sync_period = 9
    slow_step_size = 0.7496038687048765
    total_steps = 14103
    warmup_proportion = 0.1785902006326781
    min_lr =  5.219265219794206e-06

    #Callbacks
    lr_reduction_factor = 0.2979832620972566

    #Training
    batch_size = 768


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

    if optimizer_type == 'ranger':
        radam = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, total_steps=total_steps, warmup_proportion=warmup_proportion, min_lr=min_lr)
        optimizer = tfa.optimizers.Lookahead(radam, sync_period=sync_period, slow_step_size=slow_step_size)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss="mse")

    df_train, df_test = train_test_split(data, test_size=0.05, shuffle=False)

    # Step 3: Initialize and fit the scaler on the wavelet-transformed training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df_train)

    # Step 4: Scale both the training and testing data using the fitted scaler
    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    # Split the data into X,y sets
    X_train, y_train = create_dataset(df_train_scaled, num_previous_intervals)
    X_test, y_test = create_test_dataset(df_test_scaled, num_previous_intervals)
    X_test_inverse, y_test_inverse = create_test_dataset(df_test, num_previous_intervals)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    #model_checkpoint = ModelCheckpoint('trained_models/formless-v2_2.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_reduction_factor, patience=3, min_lr=min_lr, verbose=1)

    # Train the model
    model.fit(X_train, y_train, epochs=2, batch_size=batch_size, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

    # Save the scaler
    #with open('trained_models/formless-v2_2_scaler.pkl', 'wb') as file:
    #    pickle.dump(scaler, file)

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
    # Reshape back to original predictions shape
    original_scale_predictions = original_scale_predictions[:, 0].reshape(predictions.shape[0], predictions.shape[1]).flatten().tolist()
    
    pred_list = []
    for i in range(X_test.shape[0]):
        # Interp between last seen value and prediction
        full_pred = np.linspace(X_test_inverse[i][-1][0], original_scale_predictions[i], 100)
        pred_list.append(full_pred)

    print(y_test_inverse)
    rmse = calculate_weighted_rmse(pred_list, y_test_inverse)

    print(f"Models RMSE: {rmse}")
    