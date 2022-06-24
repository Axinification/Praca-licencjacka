
import warnings
# import os
import pandas as pd
import numpy as np
# import math
# import datetime as dt
import matplotlib.pyplot as plt

# import statsmodels.api as sm

# import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping

# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.express as px

# from itertools import product

# from itertools import cycle
# from plotly.subplots import make_subplots

# from sklearn.metrics import (mean_squared_error, mean_absolute_error,
#                              explained_variance_score, r2_score,
#                              mean_poisson_deviance, mean_gamma_deviance,
#                              accuracy_score)
# from unittest.mock import call
from sklearn.preprocessing import MinMaxScaler

from global_vars import LOOK_BACK, PLOTTING, ROOT_PATH
warnings.filterwarnings("ignore")

"""
Description: This program uses artificial recurrent neural network
             called Long Short Term Memory (LSTM) to predict Bitcoin
             prices using Tensorflow's Keras implementtion of LSTM module
"""


# Import dataframe
input_df = pd.read_csv(ROOT_PATH)
#  print(input_df.shape)
#  print(input_df.info())

# Convert time format to datetime64
input_df_datetype = input_df.astype({"time": "datetime64"})
#  print("Null values: "input_df_datetype.isnull().values.sum()) # Should be 0
#  print("Null values: "input_df_datetype.isnull().values.any()) # Should be F

# Drawing the lag plot - https://pythontic.com/pandas/plotting/lag%20plot
# if PLOTTING:
#     plt.style.use("seaborn-darkgrid")
#     plt.figure(1, figsize=(15, 12))
#     plt.suptitle('Lag Plots', fontsize=17)
#     plt.subplot(2, 3, 1)
#     pd.plotting.lag_plot(input_df_datetype['close'], lag=1)  # Minute lag
#     plt.title('1-Minute Lag')

#     plt.subplot(2, 3, 2)
#     pd.plotting.lag_plot(input_df_datetype['close'], lag=60)  # Hourley lag
#     plt.title('1-Hour Lag')

#     plt.subplot(2, 3, 3)
#     pd.plotting.lag_plot(input_df_datetype['close'], lag=1440)  # Daily lag
#     plt.title('Daily Lag')

#     plt.subplot(2, 3, 4)
#     pd.plotting.lag_plot(input_df_datetype['close'], lag=4320)  # Daily lag
#     plt.title('3-day Lag')

#     plt.subplot(2, 3, 5)
#     pd.plotting.lag_plot(input_df_datetype['close'], lag=10080)  # Weekly lag
#     plt.title('Weekly Lag')

#     plt.subplot(2, 3, 6)
#     pd.plotting.lag_plot(input_df_datetype['close'], lag=43200)  # Month lag
#     plt.title('1-Month Lag')

#     #  plt.figure(figsize=(15,12))
#     #  input_df_datetype.set_index("time").close
#     #  .plot(figsize=(24,7), title="Bitcoin Weighted Price")

#     plt.legend()

# Add 'date' column to the dataframe
input_df_datetype['date'] = pd.to_datetime(input_df_datetype['time'],
                                           unit='s').dt.date

# Group dataframe by date
group = input_df_datetype.groupby('date')

# Get mean closing value for each date
closing_price_groupby_date = group['close'].mean()
# print(closing_price_groupby_date.info())

# Set prediction days
prediction_days = 60
data_breakpoint = len(closing_price_groupby_date)-prediction_days

# Create train group
df_train = closing_price_groupby_date[:data_breakpoint].values.reshape(-1, 1)
#  print(df_train.shape)

# Create test group
df_test = closing_price_groupby_date[data_breakpoint:].values.reshape(-1, 1)
#  print(df_test.shape)

# Apply MinMax function and fit transform to scaler
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
#  scaler_train = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler_train.fit_transform(df_train)
#  scaler_test = MinMaxScaler(feature_range=(0,1))
scaled_test = scaler_test.fit_transform(df_test)

#  Plotting the training values
# if PLOTTING:
#     fig, ax = plt.subplots(1, figsize=(13, 7))
#     ax.plot(df_train, label='Train', linewidth=2)
#     ax.set_ylabel('Price USD', fontsize=14)
#     ax.set_title('', fontsize=16)
#     ax.legend(loc='best', fontsize=16)


def dataset_generator_lstm(dataset, look_back=LOOK_BACK):
    """
    Function to generate datasets for LSTM.
    It takes scaled dataset and look_back argument.
    Look_back argument defines the size of a time window
    we'll look back at to predict next timestep

    Transforms [1,2,3,4,5,6,7,8,9] etc. into:

        [1,2,3,4,5], [6]

        [2,3,4,5,6], [7]

        [3,4,5,6,7], [8]

    Where previous timestamps are used to predict the subsequent time step
    """
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]

        dataX.append(window_size_x)
        # This is the label // actual Y value
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


# Create train and test datasets for LSTM from scaled data
trainX, trainY = dataset_generator_lstm(scaled_train)
testX, testY = dataset_generator_lstm(scaled_test)

# https://keras.io/api/layers/recurrent_layers/lstm/
# For LSTM I need to reshape input into a
# 3D Tensor of [samples, time steps, features]

# Samples - This is the len(dataX), or the amount of data points I have

# Time stamps - A sample contains multiple time stamps,
# that is, the width of the sliding window.

# Note here that it is distinguished from
# the sliding step of the sliding window.

# So Time Steps is equivalent to the amount of
# the time steps I run my recurrent neural network.
# Features - this is the amount of features in every time step

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# return_sequence = True - it defaults to False
# The original LSTM model is comprised of single hidden LSTM layer
# followed by a standard feedforward output layer.

# The Stacked LSTM is and extension to this model that has multiple
# hidden LSTM layers where each layer contains multiple memory cells.

# I must return return_sequence=True when stacking LSTM layers so that
# the second LSTM layer has a cmpatibile n-dimensional sequence input.

# I increase the depth of my neural network to add new
# layers of abstraction of input observation over time.

# Increasing the amount of hidden layers should give me
# more accurate predictions over time.

# More on Dropout
# https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

# I must set return_sequence to False at the last hidden layer
# trainX.shape[1] - number of time steps (5)
# trainX.shape[2] - number of features (1)


# Adding the first LSTM layer and some Dropout regularisation
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True,
               input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')


# Define callbacks to help me have more controll over the training process.
# This includes stopping training when you reach a certain accuracy/loss score
# Saving my model as a checkpoint after each epoch,
# adjusting the learning rates over time, and more


# Checkpoint path
checkpoint_path = "my_model.hdf5"

# Callback to save model
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

# Used for stopping training when the least loss is achieved
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=50,
                               restore_best_weights=True)


callbacks = [checkpoint, early_stopping]

# TODO: Read fit description
history = model.fit(trainX, trainY, epochs=300,
                    verbose=1, shuffle=False,
                    validation_data=(testX, testY), callbacks=callbacks)


if PLOTTING:
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(history.history["loss"], label='Train', linewidth=1)
    ax.plot(history.history["val_loss"], label="Test", linewidth=1)
    ax.set_ylabel('Price USD', fontsize=14)
    ax.set_title('', fontsize=16)
    ax.legend(loc='best', fontsize=16)

predicted_btc_price_test_data = model.predict(testX)
predicted_btc_price_test_data = scaler_test.inverse_transform(
                                predicted_btc_price_test_data.reshape(-1, 1))

test_actual = scaler_test.inverse_transform(testY.reshape(-1, 1))

if PLOTTING:
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(history.history["loss"], label='Train', linewidth=1)
    ax.plot(history.history["val_loss"], label="Test", linewidth=1)
    ax.set_ylabel('Price USD', fontsize=14)
    ax.set_title('', fontsize=16)
    ax.legend(loc='best', fontsize=16)

if PLOTTING:
    plt.show()
