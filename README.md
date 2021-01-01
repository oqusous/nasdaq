# Stock Price Prediction Based On Time Series Baseline, Holt-Winter, and Autoregressive-Moving Average Models

The project aims to build Univariate models to forecast NASDAQ stock index’s price using Python programming language. The data and its source are first introduced and the need for such tool is explored. The data is then processed in order to keep 5500 datapoints, converted to the appropriate data types, the target variable is set as the close stock price and then split into 1:5 testing to training datapoints. The data is then is made stationary by differencing the log of the data. Through time decomposition, it is determined that the data has a strong trend and weak seasonality.<br>

The data is then modeled using Average, Naïve, Simple Exponential Smoothening, Holt’s linear, Holt-Winter Seasonal, ARMA and ARIMA methods. At the end of experimenting with each model, the best one-step forecast is determined to be Holt Linear method and the best h-step predictor is ARIMA with order (3,1,4).

The data is obtained from http://macrotrends.net/ and it consists of the daily market closing, market opening, highest and lowest price of the day and the volume of shares traded on the day. The target will be to predict the closing price of IXIC. The first datapoint is taken dated “2002-07-01” and the last is “2020-11-27” (format = Year-month-day). 80% of the data will be used as a model training dataset (2002-07-01 – 2017-03-23) and 20% of the data will bused as a testing dataset for the trained models (2017-03-24 – 2020-11-27)<br>.

## NASDAQ - Original Data (Train Set)
![GitHub Logo](/Images/NASDAQ Stock Price at COB vs Date.png)
