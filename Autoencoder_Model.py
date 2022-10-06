# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:33:49 2022

@author: ZAKY-PC
"""
import time
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 



# 1. load dataset
ticker = '%5EJKSE'
period1 = int(time.mktime(datetime.datetime(2003, 1, 1, 23,59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2022, 9, 1, 23,59).timetuple()))
interval = '1d' # id, 1mo

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
print(query_string)

# 2. set into dataframe
df = pd.read_csv(query_string)
df['Date'] = pd.to_datetime(df['Date'], format = ("%Y/%m/%d"))
df = df.set_index('Date')
print(df.head())  

# 3. check for nan value and remove
print(df.isna().sum().sum())
df = df.dropna()

# 4.plot and check the distribution
df[['Open', 'High', 'Low', 'Close', 'Adj Close','Volume']].plot(subplots=True, grid = True, figsize = (15,10))
df.describe()

# 5. create new features:- technical analysis indicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel, KeltnerChannel
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator, KAMAIndicator,ROCIndicator, PercentageVolumeOscillator, PercentagePriceOscillator
from ta.trend import *
from ta.volume import AccDistIndexIndicator, ChaikinMoneyFlowIndicator
# Trend
df['EMA'] = ema_indicator(df['Close'], window =200) #  Exponential Moving average (EMA)
df['MACD'] = MACD(df['Close'], window_slow=26, window_fast= 12, window_sign= 9).macd()  # Moving Average Convergence/DIvergence (MACD)
df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
df['SMA'] = sma_indicator(df['Close'], window = 12)
df['STC'] = stc(df['Close'], window_slow=50, window_fast=23, cycle=10, smooth1=3,smooth2=3) # Schaff Trend Cycle (STC)
df['AroonIndicator'] = AroonIndicator(df['Close'], window=25).aroon_indicator()
df['CCIIndicator'] = CCIIndicator(df['High'],df['Low'], df['Close']).cci() 
df['DPOIndicator'] = DPOIndicator(df['Close'], window=20).dpo() 
df['IchimokuIndicator'] = IchimokuIndicator(df['High'],df['Low']).ichimoku_base_line() 
df['KSTIndicator'] = KSTIndicator(df['Close']).kst() 
df['MassIndex'] = MassIndex(df['High'],df['Low']).mass_index() 
df['PSARIndicator'] = PSARIndicator(df['High'], df['Low'], df['Close']).psar()
df['TRIXIndicator'] = TRIXIndicator(df['Close']).trix() 
df['VortexIndicator'] = VortexIndicator(df['High'], df['Low'], df['Close']).vortex_indicator_diff()
df['WMAIndicator'] = WMAIndicator(df['Close']).wma() 
df['aroon_down'] = aroon_down(df['Close'])
df['aroon_up'] = aroon_up(df['Close'])
df['cci'] = cci(df['High'], df['Low'], df['Close'])
df['dpo'] = dpo(df['Close'])
df['ichimoku_base_line'] = ichimoku_base_line(df['High'], df['Low'])
df['kst'] = kst(df['Close'])
df['kst_sig'] = kst_sig(df['Close'])

# Volatility 
df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range() # Average True Range 
df['upper_bands'] = BollingerBands(df['Close']).bollinger_hband() # BollingerBands upper
df['lower_bands'] = BollingerBands(df['Close']).bollinger_lband() # BollingerBands lower
df['MA_bands'] = BollingerBands(df['Close']).bollinger_mavg() # BollingerBands moving avg
df['DC'] = DonchianChannel(df['High'], df['Low'], df['Close'], window=55).donchian_channel_mband() # Donchian Channel mid
df['DonchianChannel'] = DonchianChannel(df['High'], df['Low'], df['Close']).donchian_channel_pband()
df['KeltnerChannel'] = KeltnerChannel(df['High'], df['Low'], df['Close']).keltner_channel_pband()
# Volume
df['ADII'] = AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()  # Accumulation/Distribution Index 
df['CMFI'] = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'],df['Volume'], window=20).chaikin_money_flow() 

# Momentum
df['RSI'] = RSIIndicator( df['Close'], window=14).rsi() # Relative Strength Index (RSI)
df['Momentum Indicators'] = AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
df['KAMAIndicator'] = KAMAIndicator(df['Close'], window=10).kama()
df['Per_PriceOscillator'] =  PercentagePriceOscillator(df['Close'], window_slow = 26, window_fast= 12).ppo()
df['Per_VolOscillator'] = PercentageVolumeOscillator(df['Volume']).pvo()
df['ROC'] = ROCIndicator(df['Close']).roc() 

# Plot the chart
df.plot(subplots=True, grid = True, figsize = (15,10))

# 6. Evaluate the correlation between features
def find_corr(df):
    import seaborn as sns
    plt.figure(figsize=(20,10))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    
    heatmap = sns.heatmap(df.corr(method='pearson'), mask=mask, vmin=-1, vmax=1, annot=False, cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
    
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(df.corr(method='pearson')[['Close']].sort_values(by='Close', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Close Price', fontdict={'fontsize':18}, pad=16);
find_corr(df)

# df.plot(subplots=True, grid = True, figsize = (15,10))

# ------------------------------- Autoencoder ------------------------------------------#
import tensorflow as tf
seed = 64
tf.random.set_seed(seed)
np.random.seed(seed)
from  tensorflow.keras.layers import Activation, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras import initializers, regularizers

# 6. Scale the data using MinMaxscaler
df =  df[(df.index >= '2003-10-20') & (df.index <= '2022-09-01')]
x_data = df.drop(['Close'], axis = 1)
y_data = df['Close']

# 7. Scale the dataset
from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler()
x_data_scale = x_scaler.fit_transform(x_data)

# ----------------- Model --------------------- #
encoding_dim = 4 # The number of features to reduced into
input_dim = x_data.shape[1]

# ----------------- Input --------------------- #
input_layer = Input(shape=(input_dim,), name='Input_layer')
# Encoder
encoded = Dense(20, activation=tf.keras.layers.LeakyReLU(), name = 'Encoder_1')(input_layer)
encoded = Dense(10, activation=tf.keras.layers.LeakyReLU(), name = 'Encoder_2')(encoded)

Bottleneck = Dense(encoding_dim, activation='relu', name = 'Bottleneck')(encoded)

# Decoder
decoded = Dense(10, activation=tf.keras.layers.LeakyReLU(), name = 'Dencoder_1')(Bottleneck)
decoded = Dense(20, activation=tf.keras.layers.LeakyReLU(), name = 'Dencoder_2')(decoded)

# -------------- Output ----------------------- #
output_layer = Dense(input_dim, activation='linear', name='Output_layer')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the autoencoder model
autoencoder.compile(optimizer=Adam(),
                    loss='mse')
# Fit to train set, validate with dev set and save to hist_auto for plotting purposes
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=1e-3,
                                     patience=20,
                                     verbose=0,
                                     mode='min')]

hist_auto = autoencoder.fit(x_data_scale, x_data_scale,
                epochs=1000,
                batch_size=256,
                callbacks=my_callbacks,
                validation_split=0.3
                )

autoencoder.summary()

# Summarize history for loss
plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Create a separate model (encoder) in order to make encodings (first part of the autoencoder model)
encoder_model = Model(input_layer, Bottleneck)

# Encode and decode our test set (compare them vizually just to get a first insight of the autoencoder's performance)
data_encoder = encoder_model.predict(x_data_scale)

from sklearn.metrics import r2_score

def plot_scattered(true,pred):
    plt.figure(figsize=(10,10))
    r2="%.2f"%(r2_score(true[:,0],pred[:,0]))
    x = true[:,0].flatten()
    y = pred[:,0].flatten()
    m,b = np.polyfit(x,y,1)
    plt.scatter(x,y)
    plt.plot(x , m*x+b, 'black')
    plt.xlabel('Measured Scaled')
    plt.ylabel('Predicted Scaled')
    # plt.text(0,0,r'$R^2:{}$'.format(r2), fontsize=14)
    return plt.plot()

plot_scattered(x_data_scale,data_encoder)

close_data = y_data
df_encoder = pd.DataFrame(data_encoder, columns = ['attribute_1','attribute_2','attribute_3','attribute_4'])
close_data = y_data.rename_axis('Date').reset_index()
df_encoder['Close'] = close_data['Close']
df_encoder['Date'] = close_data['Date']

find_corr(df_encoder)

# Save the data into csv file
# df_encoder.to_csv('new_attributes.csv')