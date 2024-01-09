# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc


# Step 1: Import necessary libraries and modules
import os
import random
import time
from typing import Type, Tuple
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import template
import argparse
import traceback
import bittensor as bt
import numpy as np
import pandas as pd
from vali_config import ValiConfig
import pickle


def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    parser.add_argument("--base_model", type=str, default="model_v4_1", help="Choose the base model you want to run (if youre not using a custom one).")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


def get_model_dir(model):
    return ValiConfig.BASE_DIR + model


# Main takes the config and starts the miner.
def main( config ):
   
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    def tf_blacklist_fn(synapse: template.protocol.TrainingForward) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def tf_priority_fn(synapse: template.protocol.TrainingForward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def training_f( synapse: template.protocol.TrainingForward ) -> template.protocol.TrainingForward:
        bt.logging.debug(f'received tf')
        predictions = np.array([random.uniform(0.499, 0.501) for i in range(0, synapse.prediction_size)])
        synapse.predictions = bt.tensor(predictions)
        bt.logging.debug(f'sending tf with length {len(predictions)}')
        return synapse

    def tb_blacklist_fn( synapse: template.protocol.TrainingBackward ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def tb_priority_fn( synapse: template.protocol.TrainingBackward ) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def training_b( synapse: template.protocol.TrainingBackward ) -> template.protocol.TrainingBackward:
        bt.logging.debug(f'received lb with length {len(synapse.samples.numpy())}')
        synapse.received = True
        return synapse

    def lf_blacklist_fn(synapse: template.protocol.LiveForward) -> Tuple[bool, str]:
        bt.logging.debug("got to blacklisting lf")
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def lf_priority_fn(synapse: template.protocol.LiveForward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def live_f(synapse: template.protocol.LiveForward) -> template.protocol.LiveForward:
        # Load our model
        model = tf.keras.models.load_model(f'{ValiConfig.BASE_DIR}/mining_models/hyper_model.h5', compile=False)
        # Load our scaler
        with open(f'{ValiConfig.BASE_DIR}/mining_models/hyper_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        bt.logging.debug(f'received tf')

        synapse_data_structure = synapse.samples.numpy()

        print(synapse)
        bt.logging.debug(f"features in synapse [{len(synapse_data_structure)}],"
                         f" length of synapse ds [{len(synapse_data_structure[0])}]")
        data_structure = synapse_data_structure.T[-601:, :].T
        bt.logging.debug(f"length of windowed ds [{len(data_structure[0])}]")

        data_array = data_structure.T

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

        df = add_technical_indicators(data_df)
        # remove all NaN values
        df.dropna(inplace=True)
        df.drop('timestamps', axis=1, inplace=True)
        df.drop('date', axis=1, inplace=True)

        data  = df.values

        num_features = data.shape[1]

        data_scaled = scaler.transform(data)

        # Evaluate the model
        predictions = model.predict(data_scaled[-57:].reshape([1, 57, num_features]))

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

        synapse.predictions = bt.tensor(np.array(predicted_closes))
        
        bt.logging.debug(f'sending tf with length {len(predicted_closes)}')
        return synapse

    def lb_blacklist_fn(synapse: template.protocol.LiveBackward) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def lb_priority_fn(synapse: template.protocol.LiveBackward) -> float:
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def live_b(synapse: template.protocol.LiveBackward) -> template.protocol.LiveBackward:
        bt.logging.debug(f'received lb with length {len(synapse.samples.numpy())}')
        synapse.received = True
        return synapse

    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon( wallet = wallet, port=config.axon.port, external_port=config.axon.external_port)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn = training_f,
        blacklist_fn = tf_blacklist_fn,
        priority_fn = tf_priority_fn,
    )
    axon.attach(
        forward_fn = training_b,
        blacklist_fn = tb_blacklist_fn,
        priority_fn = tb_priority_fn,
    )
    axon.attach(
        forward_fn = live_f,
        blacklist_fn = lf_blacklist_fn,
        priority_fn = lf_priority_fn,
    )
    axon.attach(
        forward_fn = live_b,
        blacklist_fn = lb_blacklist_fn,
        priority_fn = lb_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving attached axons on network:"
                    f" {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve(netuid = config.netuid, subtensor = subtensor )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 6: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log =  (f'Step:{step} | '\
                        f'Block:{metagraph.block.item()} | '\
                        f'Stake:{metagraph.S[my_subnet_uid]} | '\
                        f'Rank:{metagraph.R[my_subnet_uid]} | '\
                        f'Trust:{metagraph.T[my_subnet_uid]} | '\
                        f'Consensus:{metagraph.C[my_subnet_uid] } | '\
                        f'Incentive:{metagraph.I[my_subnet_uid]} | '\
                        f'Emission:{metagraph.E[my_subnet_uid]}')
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Miner killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main( get_config() )