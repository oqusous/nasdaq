# Stock Price Prediction Based On Time Series Baseline, Holt-Winter, and Autoregressive-Moving Average Models

The project aims to build Univariate models to forecast NASDAQ stock index’s price using Python programming language. The data and its source are first introduced and the need for such tool is explored. The data is then processed in order to keep 5500 datapoints, converted to the appropriate data types, the target variable is set as the close stock price and then split into 1:5 testing to training datapoints. The data is then is made stationary by differencing the log of the data. Through time decomposition, it is determined that the data has a strong trend and weak seasonality.<br>

The data is then modeled using Average, Naïve, Simple Exponential Smoothening, Holt’s linear, Holt-Winter Seasonal, ARMA and ARIMA methods. At the end of experimenting with each model, the best one-step forecast is determined to be Holt Linear method and the best h-step predictor is ARIMA with order (3,1,4).

The data is obtained from http://macrotrends.net/ and it consists of the daily market closing, market opening, highest and lowest price of the day and the volume of shares traded on the day. The target will be to predict the closing price of IXIC. The first datapoint is taken dated “2002-07-01” and the last is “2020-11-27” (format = Year-month-day). 80% of the data will be used as a model training dataset (2002-07-01 – 2017-03-23) and 20% of the data will bused as a testing dataset for the trained models (2017-03-24 – 2020-11-27)<br>.

## NASDAQ - Original Data (Train Set)
![NASDAQ - Original Data (Train Set)](Images/NASDAQ_Stock_Price_at_COB_vs_Date.png)

![Train Set ACF plot](Images/ACF_plot_with_20_lags_for_NADAQ_Stock_Price.png)

![Histogram of NASDAQ stock price](Images/Histogram_of_COB_NASDAQ_stock_price.png)

ADF Statistic: 0.2751
p-value: 0.9761

## NASDAQ - Differenced Log of Data

![NASDAQ - Differenced Log of Data](Images/Differenced_Log_of_NASDAQ_Stock_Price_at_COB.png)

![Differenced Log of Data Train Set ACF plot](Images/NASDAQ_Differenced_Log_of_COB_NASDAQ_price_ACF_plot_with_20_lags.png)

![Histogram of differenced Log of COB NASDAQ stock price](Images/Histogram_of_differenced_Log_of_COB_NASDAQ_stock_price.png)

## NASDAQ - Time Decomposition

![Decomposed data](Images/tdecomp.png)

The strength of Trend for the NASDAQ stock price set is 0.9676

The strength of seasonality for the NASDAQ stock price set is 0.2024

## NASDAQ Baseline Models

![Average](Images/Average_Method_v2.png)
![Naive](Images/Naive_Method_v2.png)
![Drift](Images/Drift_Method_v2.png)
![SES alpha = 0.4](Images/SES_alpha=_0.4_Method_v2.png)
![SES alpha = 0.8](Images/SES_alpha=_0.8_Method_v2.png)

## Holt Linear and Holt-Winter Seasonal Methods

![Holt Linear Method (HLM)](Images/hml.png)
![Holt-Winter Seasonal Method (HWSM)](Images/hws.png)

![Residual ACF plots for Average, Naive, Drift, SES, HLM,HWSM](Images/NASDAQ.png)

## ARMA and ARIMA

![GPAC table](Images/gpac.png)
![ACF PCAF](Images/Picture1.png)
![grid search](Images/Picture2.png)
![root cancellation](Images/Picture3.png)

![ARMA(1,2) and ARMA(2,1) One Step Prediction](Images/arma2112ored.png)
![ARMA(1,2) H Step Forecast](Images/ARMA_12.png)
![ARMA(2,1) H Step Forecast](Images/ARMA_21.png)



