# Stock Trend and Price Prediction

This project aims to predict the Trend and Stock prices of Nifty50 stocks using a Long Short-Term Memory (LSTM) algorithm. The project includes a web application built with Streamlit for an interactive user interface.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Stock Price Prediction project leverages historical stock data from the Yahoo Finance library to train an LSTM model. The LSTM algorithm is a type of recurrent neural network known for its ability to process and predict sequential data. The project includes a web application developed with Streamlit, which provides an easy-to-use interface for users to interact with the model and visualize the predicted stock prices.

I have 2 tabs in the web app , the first tab(home page) has the Information about "Nifty50 Index" trend and future trend prediction.
The second tab , shows the information of all the "50 stocks" tredn  in Nifty .

The ```nifty.py``` file is for the First Tab , the ```Nifty_stocks.py``` for second tab.I have saved the LSTM models for both the tabs in ```model5.h5``` and 
```nifty_model.5``` , and MinMaxScaler in ```scaler.pkl```.

## Installation

To run the Stock Price Prediction project locally, follow these steps:

1. Clone the GitHub repository:
   ```
   git clone https://github.com/your-username/stock-price-prediction.git
   ```

2. Navigate to the project directory:
   ```
   cd stock-price-prediction
   ```

3. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the web application, run the following command in your terminal:

```
streamlit run app.py
```


### You can find the deployed version of the Streamlit app [here]( https://stock-trend-and-price-prediction-d78m8nk9rj.streamlit.app).


Once the application is running, you can access it by opening your browser and entering the URL displayed in the terminal.

## Data

The project uses historical stock price data of Nifty50 obtained from the Yahoo Finance library. The data is preprocessed to prepare it for training the LSTM model. The dataset consists of features such as Open, High, Low, Close, and Volume. From this i have considered only "Close" as feature for Prediction.You can use 
different feature or all the features.

## Model

The LSTM algorithm is used for stock price prediction due to its ability to capture long-term dependencies in sequential data. The model is implemented using a deep learning framework such as TensorFlow or Keras. The LSTM model takes historical stock data as input and predicts future stock prices based on the learned patterns and trends.

## Results

The Stock Price Prediction model provides forecasts for Nifty50 stock prices based on the trained LSTM algorithm. The accuracy of the predictions may vary and depend on various factors, including the quality of the data and the chosen hyperparameters for the LSTM model.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please submit a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to customize this README file to include specific details about your project, such as how to extend the functionality, examples of usage, or any additional acknowledgments.
