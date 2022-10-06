# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:30:57 2022

@author: ZAKY-PC
"""
import time
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# -------------- Load, Visualization and Scale ------------------ #
data_dir = 'C:/Users/ZAKY-PC/.spyder-py3/autosave/PROJECTS/Autoencoders_models/dataset/new_attributes.csv'
data = pd.read_csv(data_dir)

df = data[['attribute_4','attribute_3','Close','Date']]
df['Date'] = pd.to_datetime(df['Date'], format = ("%Y/%m/%d"))
df = df.set_index('Date') 
df = df[['Close','attribute_4','attribute_3']]
df[['Close','attribute_4','attribute_3']].plot(subplots=True, figsize = (15,10), grid = True, xlim = (df.index.min(),df.index.max()))

df_train = df[(df.index >= '2003-10-26') & (df.index <= '2021-05-01')]
df_test = df[(df.index >= '2021-05-01') & (df.index <= '2022-09-01')]

# Plot train, and test by Close price
fig, ax = plt.subplots(nrows=2,ncols=1, figsize= (18,14))
for axes in ax:
    axes.set_xlim(df.index.min(),df.index.max())
    axes.grid()
ax0 = ax[0]
ax0.plot(data.index, data['Close'],  '--', color='k') 

ax1 = ax[1]
ax1.plot(df_train.index, df_train['Close'],  '--', label='Train', color='r') 
ax1.plot(df_test.index, df_test['Close'],  '--', label='Test', color='b') 
ax1.legend()

x_train_unscaled = df_train #.drop(['Close'], axis = 1)
y_train_unscaled = df_train[['Close']]

x_test_unscaled = df_test # .drop(['Close'], axis = 1)
y_test_unscaled = df_test[['Close']]

# Scale the features
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

x_train = x_scaler.fit_transform(x_train_unscaled)
x_test =  x_scaler.transform(x_test_unscaled)

y_train = y_scaler.fit_transform(y_train_unscaled)
y_test = y_scaler.transform(y_test_unscaled)

# ----------- Generate 3D dataset and model -----------------------#
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, mean_absolute_percentage_error
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, GRU, Dropout, Bidirectional, BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras import initializers, regularizers

win_length = 7
batch_size = 32
num_features = len(df.columns)

train_generator = TimeseriesGenerator(x_train, y_train, length = win_length, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(x_test, y_test, length = win_length, sampling_rate=1, batch_size=batch_size)

# ----------------- Modeling -----------------------------------#
# Set random seed for as reproducible results as possible
tf.random.set_seed(42)

# create LSTM model
EPOCHS = 1000

i = Input(shape=(win_length,num_features), name = 'Input_layer')

# #### Model 2 #####
x = LSTM(400, return_sequences=(True), name = 'LSTM_1')(i)
x = LSTM(350, name = 'LSTM_2')(x)
x = Dense(1, activation='linear', name = 'Output_layer')(x)

model = Model(i, x)
print(model.summary())
model.compile(loss= 'mse', 
              optimizer= 'adam', 
              metrics = [tf.metrics.MeanAbsoluteError()]
              ) 

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=100,
                                      verbose=0,
                                      mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
    tf.keras.callbacks.History(),
] 

r = model.fit_generator(train_generator, 
                        validation_data=test_generator,
                        callbacks= my_callbacks,
                        epochs=EPOCHS,
                        shuffle=False)

# plot loss per iteration
plt.figure()
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend();

model.evaluate_generator(test_generator, verbose = 0)
prediction = model.predict_generator(test_generator)
print(prediction.shape[0])


df_pred = pd.DataFrame()
df_pred[['Close_pred']] = prediction
rev_trans = y_scaler.inverse_transform(df_pred)
df_result = df[(df.index >= '2021-05-17') & (df.index <= '2022-09-01')]
df_result['Close_pred'] = rev_trans

def accuracy_test(y_true,y_pred):
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    print('MAE test: ', mean_absolute_error(y_true, y_pred))
    print('RMSE test',np.sqrt(mean_squared_error(y_true, y_pred)))
    print('R2 test',r2_score(y_true, y_pred))
accuracy_test(df_result['Close'],df_result['Close_pred'])

df_result[['Close','Close_pred']].plot(figsize =(10,5), grid=True, 
                                       xlim=(df_result.index.min(),df_result.index.max()))