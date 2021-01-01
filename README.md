# nasdaq
Predict NASDAQ stock price index

The project aims to build Multivariate Univariate models to forecast NASDAQ stock index’s price using Python programming language. The data and its source are first introduced and the need for such tool is explored. The data is then processed in order to keep 5500 datapoints, converted to the appropriate data types, the target variable is set as the close stock price and then split into 1:5 testing to training datapoints. The data is then is made stationary by differencing the log of the data. Through time decomposition, it is determined that the data has a strong trend and weak seasonality.<br>

The data is then modeled using Average, Naïve, Simple Exponential Smoothening, Holt’s linear, Holt-Winter Seasonal, ARMA and ARIMA methods. At the end of experimenting with each model, the best one-step forecast is determined to be Holt Linear method and the best h-step predictor is ARIMA with order (3,1,4).


