# Stock Price Prediction Based On Time Series Baseline, Holt-Winter, and Autoregressive-Moving Average Models

The project aims to build Univariate models to forecast NASDAQ stock index’s price using Python programming language. The data and its source are first introduced and the need for such tool is explored. The data is then processed in order to keep 5500 datapoints, converted to the appropriate data types, the target variable is set as the close stock price and then split into 1:5 testing to training datapoints. The data is then is made stationary by differencing the log of the data. Through time decomposition, it is determined that the data has a strong trend and weak seasonality.<br>

The data is then modeled using Average, Naïve, Simple Exponential Smoothening, Holt’s linear, Holt-Winter Seasonal, ARMA and ARIMA methods. At the end of experimenting with each model, the best one-step forecast is determined to be Holt Linear method and the best h-step predictor is ARIMA with order (3,1,4).

The data is obtained from http://macrotrends.net/ and it consists of the daily market closing, market opening, highest and lowest price of the day and the volume of shares traded on the day. The target will be to predict the closing price of IXIC. The first datapoint is taken dated “2002-07-01” and the last is “2020-11-27” (format = Year-month-day). 80% of the data will be used as a model training dataset (2002-07-01 – 2017-03-23) and 20% of the data will bused as a testing dataset for the trained models (2017-03-24 – 2020-11-27)<br>.

## NASDAQ - Original Data (Train Set)

ARMA methods require data to be made stationary in as the AR (auto-regressive portion of the model) is a linear method that uses its previous lags in modelling the data. This can be investigated by:

<ul>
    <li>Calculating the augmented Dickey–Fuller test (ADF) which tests if a unit root is present in the dataset The null and alternative hypothesis tests state that:</li>
        <ul>
            <li>H0: If failed to be rejected, it suggests the time-series has a unit root, meaning it is non-stationary. Assuming confidence of 95%, p-value >= 0.05.</li>
            <li>H1: The null hypothesis is rejected, it suggests the time-series do not have a unit root, meaning it is stationary. (p-value < 0.05)</li>
        </ul>
    <li>Proving the data to be normally distributed. This can be demonstrated using a histogram plot.</li>
    <li>Showing weak auto-correlation at time lags 1 and above. This can be demonstrated using an autocorrelation stem plot.</li>
</ul>

![NASDAQ - Original Data (Train Set)](Images/NASDAQ_Stock_Price_at_COB_vs_Date.png)

![Histogram of NASDAQ stock price](Images/Histogram_of_COB_NASDAQ_stock_price.png)

The skewed nature of the histogram, confirms that the data is not normally distributed. This reinforces the findings of the ADF test that the data is non-stationary.

![Train Set ACF plot](Images/ACF_plot_with_20_lags_for_NADAQ_Stock_Price.png)

The ACF plot shows a strong linear relationship between lagged values. Taking all of the above evidence into consideration, the data is considered not stationary and further processing operations will be implemented to achieve stationarity.

ADF Statistic: 0.2751
p-value: 0.9761

p-value > 0.05, we fail to reject H0. Thus data is not stationary according to ADF test.

## NASDAQ - Differenced Log of Data

The data will be transformed using log transformation followed by first order differencing to achieve stationarity*. Pandas differencing .diff() mehod is applied and a new column is created in the training dataframe.

![NASDAQ - Differenced Log of Data](Images/Differenced_Log_of_NASDAQ_Stock_Price_at_COB.png)

![Histogram of differenced Log of COB NASDAQ stock price](Images/Histogram_of_differenced_Log_of_COB_NASDAQ_stock_price.png)

![Differenced Log of Data Train Set ACF plot](Images/NASDAQ_Differenced_Log_of_COB_NASDAQ_price_ACF_plot_with_20_lags.png)

The ACF plot shows the first order differencing of Log of data has eliminated the strong linear relationship between lagged values. The PACF and ACF plots using statsmodels library show the shaded blue area which is the confidence intervals where the standard deviation is computed according to Bartlett’s formula. The plotted stems that protrude beyond the shaded area are indicators for the orders that can be used when fitting autoregressive (AR) and moving-average (MA) models. ACF plot can help determine the order of MA- here the plot shows that orders 3 and 6 may be used. The PACF plot helps determine the order of AR, which could also shows them to be at lags 3 and 6. More on this is explored later in section 6, where ARMA and ARIMA models are used.

ADF Statistic: -18.3238
p-value: 2.2630585806598424e-30

P-value is << 0.05; hence We reject H0 and adopt H1. The ADF test indicates data is stationary.

## NASDAQ - Time Decomposition

ARIMA and SARIMA are extensions of ARMA method and are used for data that has a strong trend, seasonality or both. Time decompostion is used to approximate the strengths of these in time series data.<br>

![Decomposed data](Images/tdecomp.png)

FS is 0.206 which is a low seasonality, as such seasonality is of a lower impact on the modelling methods used. On the otherhand, FT is 0.97 which means the data exhibits very strong trend. The results indicate that ARMA and ARIMA may be sufficent to model this data.

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

In an ideal model, the residuals should classify as White Noise WN; this is defined as a squence of random numbers that cannot be predicted that has mean of zero, constant variance and a normal distribution. When that is the case, the model is considered to have perfectly captured the information in the data and no further improvements can be done to it. Another property of WN is that the ACF plot for WN has a peak equal to one at lag 0 and absolute zero for all other values. The ACF plot of the residuals for all models except average method drop to values below 0.2 after the first lag and keep on decaying at further lags. This indicates the residuals have an ACF distriubtion that is close to being White Noise The degree to which residuals represent WN can be measured by calculating the Box-Pierce test's Q value and compare it to Chi-Square test. The hypothesis for the test is:<br>
- H0: Null hypothesis: The residuals are uncorrelated, hence it is white. This can be shown when Q < Qc or when the p-value is higher than the threshold.<br>
- HA Alternative hypothesis: The residuals are correlated, hence it is NOT white. This can be shown when Q > Qc or when the p-value is lower than the threshold. <br>

Furthermore, a model's "goodness" can be evaluated using the following metrics:<br>
1. Residuals mean close to zero (model is considered as unbiased)<br>
2. Residuals have a normal distribution.<br>
3. Low mean square error score for forecast and residuals.<br>
4. Zero correlation between forecast errors and actual dependant variable values.<br>
5. Ratio of variance between forecast errors and residuals is close to 1.<br>

Method | Q-Values | P-value of Q | Var Fore Er/Var Res | MSE of Residuals | Mean of Residuals | MSE of Forecast Error |
-------|----------|--------------|---------------------|------------------|-----------------| -----------------------|
71003 | 0 |	2.47 | 235 | 9.79 | 4552 |
72 | 0 | 882 | 0.39 |	0.01 | 1017 |
222 | 0 | 846 | 0.40 | 0.02 |	1017 |
72 | 0 | 583 | 0.39 | 0.013 | 603 |
72 | 0 | 144 | 0.39 | 0.004 | 115 |
230 | 0 | 314 | 0.47 | 0.003 | 168 |

## ARMA and ARIMA

![GPAC table](Images/gpac.png)
![ACF PCAF](Images/Picture1.png)
![grid search](Images/Picture2.png)
![root cancellation](Images/Picture3.png)

![ARMA(1,2) and ARMA(2,1) One Step Prediction](Images/arma2112ored.png)
![ARMA(1,2) H Step Forecast](Images/ARMA_12.png)
![ARMA(2,1) H Step Forecast](Images/ARMA_21.png)
![ARMA(2,1) residual ACF](Image/ARMA(2,1)_resiudals_plot_with_20_lags.png)

![GirdSearch ARIMA](Image/Picture4.PNG)
![ARIMA(3,1,4) One Step Prediction](Image/arIma314_pred.png)
![ARIMA(3,1,4) H Step Forecast](Images/TSA_ARIMA_314.png)
![ARMA(3,1,4) residual ACF](Image/ARIMA(3,1,4)_resiudals_plot_with_20_lags.png)

Similar to ARMA models, the ARIMA model has successfully produced white noise residuals, near zero mean and low residual MSE. However, the forecast errors evaluated using the manual loop have higher MSE when compared to to ARMA model. However, using the forecast method in SARIMAX library produces nearly identical residual properties and statistics and better h-step forecast results. Below is the ACF plot for ARIMA314 residuals.

## Conclusion

The holt-linear method performed the best out of the baseline models in both predictions and forecasting. <br>

Both ARMA orders (1,2) and (2,1) performed well in one-step prediction resulting in WN residuals and unbiased models (mean of residuals close to 0), however, the forecasting capability was weak as it produced high forecast errors. The forecast plot indicated a very small increase in value from 65.22 to 65.32 in the first few steps, then it plateaued. A comparison between the TSA library and manual loop calculation was carried out in the appendix and the forecast results were very similar for both methods of forecasting.<br>

The ARIMA(3,1,4) model also performed well in one-step prediction, again, resulting in in WN residuals and unbiased model (mean of residuals close to 0). Using the TSA library forecast method, the forecast MSE is lower than ARMA.<br>

