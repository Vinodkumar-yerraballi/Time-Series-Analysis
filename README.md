# Time Series Analysis

### What is Time Series Analysis

### Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values. While regression analysis is often employed in such a way as to test relationships between one or more different time series, this type of analysis is not usually called "time series analysis", which refers in particular to relationships between different points in time within a single series. Interrupted time series analysis is used to detect changes in the evolution of a time series from before to after some intervention which may affect the underlying variable.

# What are they steps for " Time Series Anlysis"

Data Source link is https://finance.yahoo.com/ .

      Once we taken from the data we install the standard libraries such as given below

      import pandas as pd
      import matplotlib.pyplot as plt
      import numpy as np
      from mplfinance.original_flavor import candlestick_ohlc
      import matplotlib.dates as mpdates
      plt.style.use('dark_background')

Then move to the data loading pandas

       data=pd.read_csv('filename.csv')

Then we print statstics in the data set we use function as

        data.describe()

once we analysize the stastics in the dataset move to visualization part with we use various methods to visualize the data information on the dataset.

        1) Candlestick Chart using mplfinance
        2) Line chart using the matplotlib
          plt.plot(data.Close)

Then we create a variable for the 100days stock price with mean function the code is show in the below

       max100=data.Close.rolling(100).mean()

    Then we visualize the 100 days mean value with original price
    plt.figure(figsize=(12,8))
    plt.plot(data.Close)
    plt.plot(max100,'red')

![download](https://user-images.githubusercontent.com/98636972/197584455-5b5b3537-8217-4565-a118-ba62d71a7365.png)

And next we find the 200 days mean value and the visualize it .

       max200=data.Close.rolling(200).mean()
       plt.figure(figsize=(12,8))
       plt.plot(data.Close)
       plt.plot(max100,'red')
       plt.plot(max200,'green')

![download (1)](https://user-images.githubusercontent.com/98636972/197584607-57735e98-4293-4300-95fc-aaef32830836.png)



## After that we divided the data into train and testing.

       We take 70% data for testing and 30% for training the code given below
       data_training=pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
       data_testing=pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

Once we divided the data into training and testing part we do the scalling process with MinMaxScaler,The code is given below.

           from sklearn.preprocessing import MinMaxScaler
           scaler=MinMaxScaler(feature_range=(0,1))
           data_training_array=scaler.fit_transform(data_training)

Well, afte the scalling we create to empty list such as x_train and y_train. Once the empty list creates use for loop for prediction of next 100 days, Once create the for loop then convert then x_train and y_train values in array form using np.array function

           x_train=[]
           y_train=[]

          for i in range(100,data_training_array.shape[0]):
          x_train.append(data_training_array[i-100:i])
          y_train.append(data_training_array[i,0])
          x_train,y_train=np.array(x_train),np.array(y_train)

Now the data is ready for model buliding

# Model Buliding using the kears API

We use LSTM Method for Time Series Analysis, We insall the required layers, model. Once we install we created the sequential model with 4 layers,with LSTM with units and shape, activation function add with dropout function and finally create the dense layer.the code is given below.

             model=Sequential()
            #add first layer
            model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
            model.add(Dropout(0.2))
            #add second layer
            model.add(LSTM(units=80,activation='relu',return_sequences=True))
            model.add(Dropout(0.5))
            #add thired layers
            model.add(LSTM(units=100,activation='relu',return_sequences=True))
            model.add(Dropout(0.4))
            #add fourth layer
            model.add(LSTM(units=120,activation='relu'))
            model.add(Dropout(0.8))
            #add dence layer
            model.add(Dense(units=1))

The model is ready for Compiling. the model complie with adam optimizer and loss function then model fit with x_train and y_train with 20 epochs

        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(x_train,y_train,epochs=20)

Then model is saved .h5 file, Then we create variable for last 100 days,then we add the testing data and past 100 days. and scaling the final data

              past_100_days=data_training.tail()
              final_df=past_100_days.append(data_testing,ignore_index=True)
              input_data=scaler.fit_transform(final_df)

Again we create x_test and y_test function after covert into arrays, then we do the prediction to the x_test data using the model we save the prediction values in the varible.

           x_train=[]
           y_train=[]

          for i in range(100,data_training_array.shape[0]):
          x_train.append(data_training_array[i-100:i])
          y_train.append(data_training_array[i,0])
          x_train,y_train=np.array(x_train),np.array(y_train)

          y_predicted=model.predict(x_test)

Then we find the scale value of the scaler function and divided it to get a scaler factor and the the y_test and scaler factor is muliplie to get the y_prediction values. And then visualize the y_test and prediction data using the matplotlib.

            plt.figure(figsize=(15,7))
            plt.plot(y_test,'blue',label="Original_price")
            plt.plot(y_prediction,'red',label="Predicted_price")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.show()
            
![newplot](https://user-images.githubusercontent.com/98636972/197584718-5d3d60cf-74e6-468d-a269-fc92c591e519.png)



