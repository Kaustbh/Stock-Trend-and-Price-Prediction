# importing all necessary libraries
import yfinance as yf
from datetime import date
import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense ,Dropout ,LSTM , LeakyReLU,BatchNormalization
from keras.models import Sequential
from sklearn.metrics import r2_score
import pickle
import keras
import matplotlib.pyplot as plt
import streamlit as st


# Downloading Data based on the Dates Provided
@st.cache_resource
def load_data(start_date, end_date):
    ticker = "^NSEI"  # Ticker symbol for the Nifty index
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Predicting Nifty-50 Future Trend
@st.cache_resource
def trend(user_input):
    
    model=keras.models.load_model('model5.h5')
    scaler=MinMaxScaler(feature_range=(0,1))
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime.today()

    ticker = "^NSEI"  # Ticker symbol for the Nifty index
    data3= yf.download(ticker, start=start_date, end=end_date)
    df_close=data3.filter(['Close'])
    df_close=df_close.values
    df_close_scaled=scaler.fit_transform(df_close)

    fut_inp=df_close_scaled[-100:]
    fut_inp=fut_inp.reshape(1,-1)
    tem_inp = list(fut_inp)
    # print(fut_inp.shape)
    tem_inp=tem_inp[0].tolist()

        
    lst_output = []
    n_steps = 100
    i=0
    # For Predicting n days prices using previous n_steps
    while(i<user_input):

        if len(tem_inp) > 100:
            fut_inp = np.array(tem_inp[1:])
            fut_inp = fut_inp.reshape(1, -1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tem_inp.extend(yhat[0].tolist())
            tem_inp = tem_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1

        else:
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tem_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    #Creating a dummy plane to plot graph one after another
    plot_new = np.arange(1, 101)
    plot_pred = np.arange(101, 101 + len(lst_output))
    
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    ax.plot(plot_new, scaler.inverse_transform(df_close_scaled[-100:]))
    ax.plot(plot_pred, scaler.inverse_transform(lst_output))
    ax.legend(["Last 100 days", "Predicted {0} days".format(user_input)],loc="lower right")
    # Display the plot in Streamlit
    st.pyplot(fig)
   


    df_new=df_close_scaled.tolist()
    df_new.extend(lst_output)
    st.markdown(f"<span style='color: white; font-size: 40px;'> Smooth Curve </span>", unsafe_allow_html=True)
    fig2, ax = plt.subplots()
    ax.plot(df_new[3210:],'r')
    st.pyplot(fig2)



def niftyindex():
 
    

    
    st.markdown("<h1 style='color: red;font-family: serif '>Welcome to the Home Page! \n</h1>", unsafe_allow_html=True)
    st.markdown("<span style='color:darkslategray;font-size:25px;'>(For good visibility change the app theme to dark)</span>",unsafe_allow_html=True)
    st.markdown("<h2 style='color: 	lightseagreen;'>Nifty-50 Trend</h2>", unsafe_allow_html=True)

    min_start_date = date(2010, 1, 1)
    max_date = date.today()
    start_date = st.date_input('Start Date', min_value=min_start_date, max_value=max_date, value=date(2010, 1, 1))
    end_date = st.date_input('End Date', max_value=max_date)

    # Call the load_data function
    data = load_data(start_date, end_date)
    
    st.header('Data Statistics')
    bar = st.slider('Select data Head size', 5, len(data), 5, 5, key='data_head_size')
   
   
    st.success(f'Showing data head of size {bar}')
    st.table(data.tail(bar))
    st.success("Data Correlation")
    st.table(data.corr())
    
    # Plotting the Data
    @st.cache_resource
    def plot_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name='Stock Open',line=dict(color='aqua', width=4)))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Close',line=dict(color='red', width=4)))
        fig.layout.update(title="<span style='font-size: 30px; text-align:center;' >Time Series Data</span>", xaxis_rangeslider_visible=True, yaxis_title='Stock Price (USD per Shares)')
        st.plotly_chart(fig)

    plot_data()

    # Loading the Trained Model
    model = keras.models.load_model('model5.h5')
   
    # Load the scaler object from the file
    with open('scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)
    
    
    # Previous Steps
    n=100
    
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime.today()

    # Ticker symbol for the Nifty index
    ticker = "^NSEI"  
    data2 = yf.download(ticker, start=start_date, end=end_date)
    new_df=data2.filter(['Close'])
    last_n_days=new_df[-n:].values
    last_n_days=scaler.transform(last_n_days)
    
    X_test=[]
    X_test.append(last_n_days)
    X_test=np.array(X_test)
    
    Y_pred=model.predict(X_test)
    Y_pred=scaler.inverse_transform(Y_pred)
    
    predicted_price = Y_pred[0][0]  

    st.markdown(f"<span style='color:aqua; font-size: 40px; text-align:center;'>Predicting Next 'n' days trend using Previous 100 days </span>", unsafe_allow_html=True)
    
    user_input = st.number_input('Enter a number',10,365)
    
    # Calling the trend function
    a=trend(user_input)

    st.markdown("## Predict Tomorrow's Price based on 100 previous days:")
    submit = st.button('Predict', key='predict_button', help='Click to make a prediction', use_container_width=True)
    
    
    if submit:
        st.markdown(f"<span style='color: white; font-size: 40px;'>Stock Price Prediction : </span><span style='color: aqua; font-size: 40px;'>{predicted_price:.2f}</span>", unsafe_allow_html=True)
    