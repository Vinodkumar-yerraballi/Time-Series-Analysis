from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
from keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime

start='2005-01-01'
end='2022-10-18'
st.title("Time Series Anlysis")
user_input=st.text_input("Enter stock","AAPL")
df=data.DataReader(user_input,'yahoo',start,end)
st.subheader("Data from 2005 to 2022")
st.write(df.describe())

st.subheader("To visualize the close price")

#To visualize the close price in the data frimae
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("To visualize the last 100 days mean ")

#To view the last 100 values mean 
max100=df.Close.rolling(100).mean()
#Let's visualize the it
fig2=plt.figure(figsize=(12,8))
plt.plot(df.Close)
plt.plot(max100,'red')
st.pyplot(fig2)
st.subheader("To visualize the last 200 days mean")
#To view the last 200 days mean
max200=df.Close.rolling(200).mean()
fig3=plt.figure(figsize=(12,8))
plt.plot(df.Close)
plt.plot(max100,'red')
plt.plot(max200,'green')
st.pyplot(fig3)

# We take the 70 training data 
# and 30% per testing data 
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
x_train=[]
y_train=[]
#Scalling the values using the MinMaxScaler 

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)
#create for loop for feature prediction for next 100 days
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

model=load_model('LSTM_1.h5')
# Last 100  days information

past_100_days=data_training.tail()
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)

scale=scaler.scale_
scale_factor=1/scale[0]
y_prediction=y_predicted*scale_factor
y_test=y_test*scale_factor

fig4=plt.figure(figsize=(15,7))
plt.plot(y_test,'blue',label="Original_price")
plt.plot(y_prediction,'red',label="Predicted_price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig4)
y_pred=y_prediction.reshape(-1,)
data1=pd.read_csv('AMZN.csv')
data2=data1["Date"]
st.set_option('deprecation.showPyplotGlobalUse', False)
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=data2, y=y_test, name="Test", mode="lines"))
fig5.add_trace(go.Scatter(x=data2, y=y_pred, name="Prediction", mode="lines"))
fig5.update_layout(
    title="Amazon stock prices", xaxis_title="Date", yaxis_title="Close"
)
st.pyplot(fig5.show())