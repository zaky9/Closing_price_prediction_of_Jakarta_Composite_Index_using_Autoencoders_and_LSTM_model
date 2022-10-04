# Closing price prediction of Jakarta Composite Index using Autoencoders and LSTM model 
By Zaky Riyadi

<p align="justify"> 
In this repo, Im trying to predict the Closing price of **Jakarta Composite Index (JKSE)** from multiple **historical data** and **technical analysis Index**. Predicting the Closing price of JKSE is part of forcasting timeseries problem, where the every observations are time-dependent and the output data are continous. Predicting the stock price is  a complex task. There are mainly 4 components that we need to look at in time series problems: Seasonality, Trend, cycles and Irregular components. Therefore, using many observation data such as Historical and technical indicator data may helps to forcast the future closing price. </p>

## Autoencoders

<p align="justify">
Using too many observation/data to forcast may resulted in **dimensionality problem**. Therefore, in this study Autoencoders is used as Feature extraction for dimentionallity reduction by transforming the data into lower dimensions that can help to prevent overfitting. **AutoEncoder** is an unsupervised ANN that tries to encode the data by compressing it into the lower dimensions (also known as bottleneck layer) and then decoding the data to reconstruct the original input. The bottleneck layer holds the compressed representation of the input data. </p>

## LSTM

## Data Visualization 
The input data we are using are historical data which are Open price, High Price, Low price, Close price, Adj Close and Volume. 
<img width="500" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_GRU_model/blob/9d402955dee2d30613c01bc9eb281a0730398b3e/Images/Picture1.png">

### Result page
![](screenshots/result.PNG)

