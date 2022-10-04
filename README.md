# Closing price prediction of Jakarta Composite Index using Autoencoders and LSTM model 
By Zaky Riyadi

<p align="justify"> 
In this repo, Im demonstrating how to predict the Closing price of the Jakarta Composite Index (JKSE) from multiple historical and technical indicators data. Predicting the Closing price of JKSE is part of the forecasting time-series problem, where every observation is time-dependent, and the output data are continuous. There are mainly 4 components that we need to look at four components in time series problems: Seasonality, Trend, cycles and Irregular. Therefore, using many observation data such as Historical and technical indicators may help forecast the closing price.</p>

## Autoencoders

<p align="justify">
Using too many features to forecast the Closing price may result in a dimensionality problem, where the model starts to overfit. Overfitting is when the model learns too well from the training dataset and fails to generalize well for unseen real-world data (testing dataset). Therefore, in this study, Autoencoders is used as a Feature extraction method for dimensionality reduction by transforming the data into lower dimensions. Autoencoder is an unsupervised artificial neural network that tries to encode the data by compressing it into the lower dimensions (also known as the bottleneck layer) and then decoding the data to reconstruct the original input. The bottleneck layer holds the compressed representation of the input data or the data that has reduced dimension. Shown in figure 1 is the diagram of Autoencoder. </p>

<center><img width="500" alt="rnn_step_forward" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/f39ed904d5e398172dd933837c8efb03a8073bf9/Images/autoencoders.png"></center>

Figure 1: generalization of Autoencoder. [[ref]](https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/)

## LSTM

<p align="justify">
Long Short-Term Memory or LSTM is a Recurrent Neural Network (RNN) commonly used for time series prediction. The advantage of using RNN is allowing previous outputs to be used as inputs while having hidden states which enables RNN to remember previous information. RNN has memory, which helps to remember all the information about what has been calculated. Therefore it is very advantageous for solving problems where observation depends on a sequence, such as in time series and NLP. However, RNN has many limitations, including: </p>

1. Short-term memory: forgetting the earliest information when moving to later ones), 
2. Vanishing gradient: the gradient becomes very small, preventing the neural network from stopping learning. (gradient shrinks as it back propagates through time)
3. Exploding gradient: the network assigns unreasonably high importance to the weights.


LSTM encounter these issues by having a cell state (or long-term memory) which runs through the chain with only linear interaction, keeping information flow unchanged. Every cell state has a gate mechanism (Input, output, and forget gate) that decides whether to keep or omit information. It is a way to pass the information selectively that consists of the sigmoid layer, hyperbolic tangent layer, and point-wise multiplication operation. To understand LSTM further, you can read the original publication [here](http://www.bioinf.jku.at/publications/older/2604.pdf), or you can read a simpler version [here](https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359) 


<img width="500" alt="rnn_step_forward" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/f39ed904d5e398172dd933837c8efb03a8073bf9/Images/1_ULozye1lfd-dS9RSwndZdw.png">

Figure 2: LSTM cell [[ref]](https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359)


## Loading data and Visualization 

Now, once we know what are Autoencoders and LSTM is and why it is used, we can finally start the analysis. 
1. First of we will download the dataset from yahoo finance. We are using the historical data ( Open price, High Price, Low price, Close price, Adj Close and Volume) of Jakarta Composite Index (JKSE) from 01/01/2003 to 01/09/2022 (dd/mm/yyyy). Once we download the dataset, we can convert from csv to panda daraframe and make the Date as the index.
2. Check for the outliars by displaying all of the features and observe the statistical data from univeriate analysis.
<img width="500" alt="rnn_step_forward" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/Picture1.png">

Figure 3: Plot between Historical data vs date

3. Once we have removed all of the outliars data, we can start to calculate the technical analysis using Technical Analysis libary [here](https://technical-analysis-library-in-python.readthedocs.io/en/latest/). In this repo Im using 37 technical analysis indicator based on the trend, volatility, volume and momentum indicator. 
<img width="500" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/TA.png">

Figure 4: All of the calculated TA

4. Once we have generated all of necessary features that we want, lets observe the monotonic correlation between every features using Spearman correlation. Based on the correlation, there are too many redundant features that are unnecessay and can highly influence the prediction. Now lets finally use Autoencoders!

<img width="400" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/Picture3.png">

Figure 5: Spearman correlation between features

<img width="300" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/Picture4.png">

Figure 6: Features Spearman correlation with the target data (Close price)

5. But wait!, before we input our features into Autoencoders, we need to rescale  the data from the original range so that all values are within the new range of 0 and 1 by using MinMaxscaler() and additionally, we need to remove the Close price and Adj CLose price to better represent the new attributes.

<img width="500" alt="rnn_step_forward" src="">



### Result page
![](screenshots/result.PNG)

## Refferences
https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/
https://www.sciencedirect.com/science/article/pii/S2666827022000378
https://medium.datadriveninvestor.com/a-high-level-introduction-to-lstms-34f81bfa262d
https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
