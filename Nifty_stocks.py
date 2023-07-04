# importing all necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
from plotly.subplots import make_subplots
from datetime import timedelta
from datetime import date
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
import keras


# Load the Data
@st.cache_resource
def load_data(start_date, end_date,user_input):
     
    data = yf.download(user_input, start=start_date, end=end_date)
    return data

# Main Function 

def stocks():

    st.markdown("<h1 >Stock Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 >Nifty-50 all Stocks</h2>", unsafe_allow_html=True)

    ticker_symbols = ['GRASIM.NS','ITC.NS','BPCL.NS','BAJFINANCE.NS','RELIANCE.NS','SBIN.NS',
                      'HDFC.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS', 'M&M.NS', 'HINDALCO.NS',
                       'HDFCBANK.NS', 'NTPC.NS', 'ICICIBANK.NS', 'TATASTEEL.NS', 'HINDUNILVR.NS', 'BAJAJFINSV.NS',
                       'WIPRO.NS', 'COALINDIA.NS', 'INDUSINDBK.NS', 'DIVISLAB.NS', 'HDFCLIFE.NS', 'KOTAKBANK.NS',
                        'ADANIENT.NS', 'ADANIPORTS.NS', 'TITAN.NS', 'INFY.NS', 'BHARTIARTL.NS', 'TATACONSUM.NS',
                        'ASIANPAINT.NS', 'BRITANNIA.NS', 'AXISBANK.NS', 'APOLLOHOSP.NS', 'HCLTECH.NS', 'HEROMOTOCO.NS',
                        'TATAMOTORS.NS', 'SBILIFE.NS', 'LT.NS', 'TECHM.NS', 'TCS.NS', 'DRREDDY.NS', 'UPL.NS', 'NESTLEIND.NS',
                        'MARUTI.NS', 'CIPLA.NS', 'SUNPHARMA.NS', 'BAJAJ-AUTO.NS', 'POWERGRID.NS'
                      ]
    user_input = st.selectbox("Select a ticker symbol", ticker_symbols)

    
    min_start_date = date(2010, 1, 1)
    max_date = date.today()
    # max_date += timedelta(days=1) 
    start_date = st.date_input('Start Date', min_value=min_start_date, max_value=max_date, value=date(2010, 1, 1))
    end_date = st.date_input('End Date', max_value=max_date)
    
    # Stick Information 
    stock_info = yf.Ticker(user_input).info
    company_name = stock_info['longName']
    st.subheader(company_name)
    market_price = stock_info['currentPrice']
    previous_close_price = stock_info['regularMarketPreviousClose']
    st.write('Market price :', market_price)
    st.write('Previous close price : ', previous_close_price)

    # Data 

    df=load_data(start_date,end_date,user_input)

    st.header('Data Statistics')
    bar = st.slider('Select data Head size', 5, len(df), 5, 5, key='data_head_size')
   
    
    st.success(f'Showing data head of size {bar}')
    st.table(df.tail(bar))
    st.success("Data Correlation")
    st.table(df.corr())
    
    # Analysis of Data 
    st.subheader('Technical Analysis')
    infoType = st.radio(
            "Choose Technical Analysis Type",
            ('Moving Average Chart', 'Market trend')
        )
    
    if infoType == 'Moving Average Chart':
        
        # Plot -: 1
        st.subheader('Closing Price')
        fig0 = go.Figure()
        fig0.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price',line=dict(color='lime',width=4)))
        fig0.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True,
            legend=dict(x=1, y=0)
        )
        fig0.layout.update(xaxis_rangeslider_visible=True, yaxis_title='Stock Price (USD per Shares)')
        st.plotly_chart(fig0)
        
        # Plot -: 2
        st.subheader('Closing Price vs Time Chart with 100 MA')
        ma100 = df.Close.rolling(100).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100 MA',line=dict(color='red',width=4)))
        fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price',line=dict(color='lime',width=3)))
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True,
            legend=dict(x=1, y=0)
        )
        fig.layout.update( xaxis_rangeslider_visible=True, yaxis_title='Stock Price (USD per Shares)')
        st.plotly_chart(fig)

        # Plot -: 3
        fig2 = go.Figure()
        st.subheader('Closing Price vs Time Chart with 100 MA & 200MA')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig2.add_trace(go.Scatter(x=df.index, y=ma100, name='100 MA',line=dict(color='red',width=3)))
        fig2.add_trace(go.Scatter(x=df.index, y=ma200, name='200 MA',line=dict(color='yellow',width=4))) 
        fig2.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price',line=dict(color='lime',width=3)))
        fig2.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True,
            legend=dict(x=1, y=0)
        )
        fig2.layout.update(xaxis_rangeslider_visible=True, yaxis_title='Stock Price (USD per Shares)')
        st.plotly_chart(fig2)

       
    else:
       
        # Get today's date
        today = date.today()

        # Define the target date
        target_date = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')

        # Calculate the difference in years
        years_difference = today.year - target_date.year

       
        n= st.slider('Stock Trend over "n" years',1,years_difference,1)
        start = dt.datetime.today() - dt.timedelta(n * 365)
        end = dt.datetime.today()
        
        # Load the data 
        df_new = yf.download(user_input, start, end)
        
        # Ploting the Data Based on number of years 
        fig = go.Figure(
            data=go.Scatter(x=df_new.index, y=df_new['Close'],line=dict(color='red',width=5))
        )
        fig.layout.update(  xaxis_rangeslider_visible=True, yaxis_title='Stock Price (USD per Shares)')
        fig.update_layout(
            title={
                'text': "Stock Prices Over Past {0} Years".format(n),
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)

   

    # Predicting the Future Prices

    start_date = date(2010, 1, 1)
    end_date = date.today()
    df_new2 = yf.download(user_input, start=start_date, end=end_date)
    df3=df_new2.filter(['Close'])
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df3).reshape(-1,1))

    datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
    st.write("<span style='font-size:30px;'>Using Previous 100 days for Prediction</span>",unsafe_allow_html=True)
    datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
    x_input=df1[-100:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # Loading the Trained Model
    model=keras.models.load_model('nifty_model.h5')
    
    date1 = st.date_input("Enter the Date till you want the Predictions")

    result = st.button("Predict") 

    if result:
                    from datetime import datetime
                    my_time = datetime.min.time()
                    date1 = datetime.combine(date1, my_time)
                    
                    nDay=date1-datemax
                    nDay=nDay.days
                    
                    date_rng = pd.date_range(start=datemax, end=date1, freq='D')
                    date_rng=date_rng[1:date_rng.size]
                    lst_output=[]
                    n_steps=x_input.shape[1]
                    i=0

                    # For Predicting n days prices using previous n_steps
                     
                    while(i<nDay):
                    
                        if(len(temp_input)>n_steps):
                                #print(temp_input)
                                x_input=np.array(temp_input[1:]) 
                                # print("{} day input {}".format(i,x_input))
                                x_input=x_input.reshape(1,-1)
                                x_input = x_input.reshape((1, n_steps, 1))
                                #print(x_input)
                                yhat = model.predict(x_input, verbose=0)
                                # print("{} day output {}".format(i,yhat))
                                temp_input.extend(yhat[0].tolist())
                                temp_input=temp_input[1:]
                                #print(temp_input)
                                lst_output.extend(yhat.tolist())
                                i=i+1
                        else:
                                x_input = x_input.reshape((1, n_steps,1))
                                yhat = model.predict(x_input, verbose=0)
                                # print(yhat[0])
                                temp_input.extend(yhat[0].tolist())
                                # print(len(temp_input))
                                lst_output.extend(yhat.tolist())
                                i=i+1
                    res =scaler.inverse_transform(lst_output)
              
                    # Printing Todays predicted price
                    output = res[0]
                    curr_date=date.today()

                    st.write("*Predicted Price for Date :*", curr_date, "*is*", np.round(output[0], 2))
                    st.success('The Price is {}'.format(np.round(output[0], 2)))

                   
                    predictions=res[res.size-nDay:res.size]
                   
                    predictions=predictions.ravel()
                   

                    # Converting the Predicted Prices to DataFrame
                    @st.cache_resource
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')
                    df = pd.DataFrame(data = date_rng)
                    df['Predictions'] = predictions.tolist()
                    df.columns =['Date','Price']
                    st.write(df)
                    csv = convert_df(df)
                    st.download_button(
                        "Press to Download",
                        csv,
                        "file.csv",
                        "text/csv",
                        key='download-csv'
                    )

                    #Visualizing Predicted Prices
                    fig = go.Figure(data=go.Scatter(x=date_rng, y=predictions, line=dict(color='navy',width=5)))
                    fig.update_layout(
                        title='Predicted Prices',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Price'),
                        xaxis_tickangle=-90,
                        width=800,
                        height=500
                    )
                    fig.layout.update(  xaxis_rangeslider_visible=True, yaxis_title='Stock Price (USD per Shares)')
                    st.plotly_chart(fig)

   


    