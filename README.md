# Closing price prediction of Jakarta Composite Index using Autoencoders and LSTM model 
By Zaky Riyadi

<p align="justify"> 
In this repo, Im going to show you how we can use Autoencoders and LSTM model to predict the closing price of the Jakarta Composite Index (JKSE) from the historical and technical indicators data. Predicting or forecasting the Closing price is part of a time-series problem, where every observation is time-dependent, and the output data are continuous.
There are many technical indicators that can help us to make better decisions on whether to buy or sell the stock. Technical indicators are pattern-based signals calculated from the Historical data (Open, Closing, High, Low, and Volume) and have many purposes ranging from measuring volatility, momentum, trend and volume.

In this study, I'm going to use 37 of technical indicators with an additional of 5 historical data to help predict the closing price of JKSE. Now you may ask,
  
"Wouldn't that be too many features and may create dimensionality problems?"
 
 and I'd say
  
"yes! And that's where Autoencoder comes in"
</p>

## Why are we using Autoencoders?

<p align="justify">
Using too many features may result in a dimensionality problem, where the model starts to overfit. Overfitting is when the model learns too well from the training dataset and fails to generalize well for unseen real-world data (testing dataset). Therefore, in this study, Autoencoders is used as a Feature extraction method for dimensionality reduction by transforming the data into lower dimensions. Autoencoder is an unsupervised artificial neural network that tries to encode the data by compressing it into the lower dimensions (also known as the bottleneck layer/ cell) and then decoding the data to reconstruct the original input. The bottleneck layer holds the compressed representation of the input data or the data that has reduced dimension. Shown in Figure 1 is the diagram of  Autoencoder. </p>

<center><img width="500" alt="autoencoders" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/f39ed904d5e398172dd933837c8efb03a8073bf9/Images/autoencoders.png"></center>

Figure 1: generalization of Autoencoder. [[ref]](https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/)

## Why are we using LSTM model?

<p align="justify">
Long Short-Term Memory or LSTM is a Recurrent Neural Network (RNN) commonly used for time series prediction. The advantage of using RNN is allowing previous outputs to be used as inputs while having hidden states which enables RNN to remember previous information. RNN has memory, which helps to remember all the information about what has been calculated. Therefore it is very advantageous for solving problems where the observation depends on a sequence, such as in time-series and NLP. However, RNN has many limitations, including: </p>

1. Short-term memory: forgetting the earliest information when moving to later ones 
2. Vanishing gradient: the gradient becomes very small, preventing the neural network from stopping learning. 
3. Exploding gradient: the network assigns unreasonably high importance to the weights.

LSTM encounter these issues by having a cell state (or long-term memory) which runs through the chain with only linear interaction, keeping information flow unchanged. Every cell state has a gate mechanism (Input, output, and forget gate) that decides whether to keep or omit information. It is a way to pass the information selectively that consists of the sigmoid layer, hyperbolic tangent layer, and point-wise multiplication operation. 

You can read the original publication here to learn more about the API and the math [here](http://www.bioinf.jku.at/publications/older/2604.pdf) or you can read a simpler version [here](https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359) or if you just want to watch someone to clearly explain LSTM you can watch [here](https://www.youtube.com/watch?v=LfnrRPFhkuY).



<img width="500" alt="LSTM" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/f39ed904d5e398172dd933837c8efb03a8073bf9/Images/1_ULozye1lfd-dS9RSwndZdw.png">

Figure 2: LSTM cell [[ref]](https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359)


## Data preparation and analysis

Now, Once we get through the **concept** and **why** we are using Autoencoders and LSTM, we can finally start the analysis.
<p align="justify">
1. First of we need to get the dataset. Im extracting the historical data from yahoo finance. We can extract the data directly using python and by following Figure 3. I have specified to use the historical data range from 01/01/2003 (period 1) to 01/09/2022 (period 2). Once we load the dataset, we can convert it from CSV to Dataframe, make the Date as the index and remove any Null values. </p>

<img width="1000" alt="load_data" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/4bee905dca73ea2961ae9464a61114ee3048888c/Images/load_data.png">

Figure 3: Load the data


2. Next, check for any outliars data by displaying all of the features and observe the statistical data from univeriate analysis.

<img width="500" alt="lot between Historical data vs date" 
src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/Picture1.png">

Figure 4: Plot between Historical data vs date

3. Once we have removed all of the outliers data, we can start calculating the technical analysis using the Technical Analysis library [here](https://technical-analysis-library-in-python.readthedocs.io/en/latest/). In this study, Im using 37 technical analysis indicator based on the trend, volatility, volume and momentum indicator.

<img width="500" alt="TA calculation" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/TA.png">

Figure 5: All of the calculated TA

4. Now, let's observe the correlation between every feature using Spearman's correlation. Based on the correlation (Figure 6 & 7), there are too many redundant features that are unnecessay and can highly influence the prediction's accuracy. Therefore, let's finally use Autoencoders!

<img width="400" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/Picture3.png">

Figure 6: Spearman correlation between features

<img width="300" alt="rnn_step_forward" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/86399dab040d534406282c4430807fd6e8afe40f/Images/Picture4.png">

Figure 7: Features Spearman correlation with the target data (Close price)

5. But Hold on!, before we input our features into Autoencoders, we need to rescale the data from the original range so that all values are within the new range of 0 and 1 by using MinMaxscaler() and additionally, we need to remove the Close price to better represent the new attributes/features.

## Autoencoders

6. Once we know that our data is "clean" (meaning no outliers) and are scaled, we can start to build our Autoencoders. Shown in Figure 8 is the model summary of the Autoencoders. Here, Im using two layers of encoder and decoders. Where the first encoder reducess the dimension from 42 (Number of original features) to 20 and subsequently to 10. The bottleneck layers (or code) are the layers that we want to extract, which it is reduced into 4 attributes.

<img width="500" alt="The model summary for Autoencoders" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/871414a0151733eba8aee92ef9e884516c102a53/Images/autoencoders_model_summary.png">


Figure 8: The model summary for Autoencoders

"But wait", you may asked.

How about Decoder layers? Why are we not using it?

<p align="justify">
Decoder layers are commonly represented as a mirror with the encoder layers. Meaning the input and the output must have a similar number of dimensions. It is a common practice to have a mirror-like shape (E.g. Encoder 1 = Decoder 2, Encoder 2 = Decoder 1). Hence, if you look at the shape between encoder and decoder are similar. Additionally, the output of the decoders layer are commonly used to compress the image or want to have identical representative features as the input. However, since our objective is to reduce the dimension, we only extract the attributes from the Bottleneck layer that have decreased from 42 to 4. </p>


7. Observing the train loss and the val loss (Figure 9), they shows a good fit, where both loss decrease and stabilize at similar epoch (at epoch 26). 


<img width="300" alt="Autoencoders loss" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/2d69aa78b439af8875089fd9379a818dbeec52a1/Images/autoencoder_loss.png">

Figure 9: Loss vs epoch 

<p align="justify">
Now let's evaluate the new attributes. I have renamed the newly generated attributes into attributes 1 to 4. Shown in Figure 10 is the Pearson correlation. The result demonstrates that attributes 3 and 4 are very well correlated with the closing price, and by plotting the line graph between close, attribute 3 and attribute 4, we can see the monotonic trend between them (Figure 11). Since attributes 1 and 2 could not represent the features very well, let's remove them and only use attributes 3 and 4 to predict the closing price. </p>



<img width="300" alt="The model summary for Autoencoders" src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/871414a0151733eba8aee92ef9e884516c102a53/Images/Picture5.png">

Figure 10: Spearman correlation between attribute 1–4 to Closing price

<img width="500" alt="Line plot " src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/871414a0151733eba8aee92ef9e884516c102a53/Images/Picture6.png">

Figure 11: Line plot between Close, attribute 3 and attribute 4

## LSTM

8. Now, let's finally predict the closing price using LSTM. First, I have split the dataset into train and test sets based on the date. My training set is from 23/10/2003 to 01/05/2021, while from 01/05/2021 to 01/09/2022 is my test set. Next, we scale the features and target data using MinMaxscaler().

<img width="500" alt="Line plot " src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/871414a0151733eba8aee92ef9e884516c102a53/Images/Picture2.png">

Figure 12: plot between the training and testing dataset

9. Now let's developed the LSTM model. LSTM requires three dimensions of input. These include; the number of batch sizes, timestep and features. In this study, Im using TensorFlow's time-series generator module, where we need to specify the window length, sampling rate and batch size. You can read the documentation [here](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator)

12. Shown in Figure 13 is the model summary of the LSTM model. Im using two layer of LSTM with the neuron of 400 and 350, respectively. Other hyper-parameters include:

* Optimizer = Adam
* Learning rate = 0.001 (with callback of ReduceLROnPlateau)
* Epoch = 1000 (with callback of Earlystopping)

<img width="500" alt="Line plot " src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/0922812d69d288f76d6bfe4b0d48d9c0b1f991db/Images/LSTM%20model.png">

Figure 13: Model summary of the LSTM model

Now, observing the loss vs epoch, we can see that the validation loss was initially overfitting, but in the later epoch, it started to converge with the training loss at around 25th epoch.

<img width="500" alt="Line plot " src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/103c670d079fa99588f64d9d3a4ee8dd0c7680e3/Images/Loss_LSTM.png">

Figure 14: train loss vs Val loss

Let's observe the prediction's accuracy. Shown in Figure 15 is the line plot between the predicted and actual closing price. We can see that the predicted close price can follow closely with the actual Closing price. Looking at the accuracy scores, we get a decent score, where R^2 of 0.97, MAE of 52.8 and RMSE of 71. 

<img width="500" alt="true_close_vs_close_pred " src="https://github.com/zaky9/Closing_price_prediction_of_Jakarta_Composite_Index_using_Autoencoders_and_LSTM_model/blob/d405b769db09d7c2c9f14c9418ae1164dd74797e/Images/Picture7.png">

Figure 15: Comparing the true close vs. close prediction at test dataset

So thats it!. I realy do hope that you enjoy and learnt something from this repo!. Im trying to make more this kin of repo and also youtube video. So stay tuned!
### Result page
![](screenshots/result.PNG)

## Refferences
https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/
https://www.sciencedirect.com/science/article/pii/S2666827022000378
https://medium.datadriveninvestor.com/a-high-level-introduction-to-lstms-34f81bfa262d
https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
