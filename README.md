# Closing price prediction of Jakarta Composite Index using Autoencoders and LSTM model 
By Zaky Riyadi

<p align="justify"> 
In this repo, Im demonstrating how to predict the Closing price of the Jakarta Composite Index (JKSE) from multiple historical and technical indicators data. Predicting the Closing price of JKSE is part of the forecasting time-series problem, where every observation is time-dependent, and the output data are continuous. There are mainly 4 components that we need to look at four components in time series problems: Seasonality, Trend, cycles and Irregular. Therefore, using many observation data such as Historical and technical indicators may help forecast the closing price.</p>

## Autoencoders

<p align="justify">
Using too many features to forecast the Closing price may result in a dimensionality problem, where the model starts to overfit. Overfitting is when the model learns too well from the training dataset and fails to generalize well for unseen real-world data (testing dataset). Therefore, in this study, Autoencoders is used as a Feature extraction method for dimensionality reduction by transforming the data into lower dimensions. Autoencoder is an unsupervised artificial neural network that tries to encode the data by compressing it into the lower dimensions (also known as the bottleneck layer) and then decoding the data to reconstruct the original input. The bottleneck layer holds the compressed representation of the input data or the data that has reduced dimension. Shown in figure 1 is the diagram of Autoencoder. </p>

<center><img width="500" alt="rnn_step_forward" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/f39ed904d5e398172dd933837c8efb03a8073bf9/Images/autoencoders.png"></center>
<strong> Figure 1:</strong> generalization of Autoencoder

## LSTM
Long Short Term Memory or LSTM is a type of Reccurent Neural Network (RNN) that is commonly used for time series prediction. The advantage of using RNN is allow previous outputs to be used as inputs while having hidden states which enables for RNN to remembers previous information. RNN has memory which allows to remembers all of the information about what has been calculated. Therefore it is very adventageous for solving problems where observation depends on a sequence such as in time series and NLP. However, RNN has many limitation including: 
1. Short term memory: forgeting the earliest information when moving to later ones), 
2. Vanishing gradient: the gradient start to become very small preventing the neral network to stop learning. (gradient shrinks as it back propagates through time)
3. Exploding gradient: the network assigns unreasonably high importance to the weights.

 LSTM encounter these issues by having a cell state (or long-term memory) which runs through the chain  with only linear interaction, keeping information flow unchanged. in every cell state, it has gates mechanism that decide whether to keep or forget information. It is a way to pass the information selectively that consists of the sigmoid layer, hyperbolic tangent layer, and the point-wise multiplication operation. There are three different type of gates: Input, output, and forget gate.



## Loading data and Visualization 
The input data we are using are historical data which are Open price, High Price, Low price, Close price, Adj Close and Volume. 
<img width="500" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_GRU_model/blob/9d402955dee2d30613c01bc9eb281a0730398b3e/Images/Picture1.png">

### Result page
![](screenshots/result.PNG)

## Refferences
https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/
https://www.sciencedirect.com/science/article/pii/S2666827022000378
https://medium.datadriveninvestor.com/a-high-level-introduction-to-lstms-34f81bfa262d
https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
