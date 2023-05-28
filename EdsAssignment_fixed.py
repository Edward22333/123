# just changed the filenames so ./datasets path is in them again

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import itertools
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from dateutil.relativedelta import relativedelta
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedKFold
# from sklearn.cluster import KMeans

# Long-term monthly forecaster. Requires ./datasets/auckland_combo_2003-2023.csv
# Use this function for long-term monthly forecasts, e.g., what will the mean max temperature be in Auckland by January 2050?
# I used some code from this tutorial: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

# Parameters:
# 'feature' is the dataset column you want to forecast for, e.g., you want to predict 'Tmax(C)'
# 'dt' is the date and time you want to predict for, as a string in this format: '2050-01-01'

def monthly_forecaster(feature, dt):
    # preps dataframe
    query = pd.read_csv('./datasets/auckland_combo_2003-2023.csv')
    query = query[[feature,'Day(Local_Date)']]
    query = query.rename(columns={'Day(Local_Date)':'Date'})
    query[feature] = query[feature].replace('-',np.nan)
    query = query.fillna(query.bfill())
    query = query.set_index('Date')
    query.index = pd.to_datetime(query.index, format='%Y%m%d:%H%M')
    query = query.truncate(before='2003', after='2023')
    query[feature] = query[feature].astype('float')
    query = query.resample('MS').median()
    query_date = datetime.strptime(dt, '%Y-%m-%d')
    end_date = pd.to_datetime(query[feature].tail(1).index[0])
    start_date = pd.to_datetime(query[feature].head(1).index[0])

    if ((query_date - end_date).days < 0):
        print("\nQuery date is already in observed history. Pick a future date.")
        return

    query.plot(figsize=(20, 4), title="History", ylabel=feature)

    # generate different combinations of p, d and q triplets
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    print('\nComputing', feature, end='')
    aics = {'param':[], 'param_seasonal':[], 'score':[]}
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(query,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                print('.', end='')
                aics['param'].append(param)
                aics['param_seasonal'].append(param_seasonal)
                aics['score'].append(results.aic)
            except:
                continue

    print('\n\nCombination with best AIC score:')
    aics = pd.DataFrame(aics)
    aics = aics[aics.score > 0]
    print(aics[aics.score == min(aics.score)],'\n')
    param = aics[aics.score == min(aics.score)]['param'].iloc[0]
    param_seasonal = aics[aics.score == min(aics.score)]['param_seasonal'].iloc[0]

    # training model with optimal p d q combo
    mod = sm.tsa.statespace.SARIMAX(query,
                                    order=param,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])

    # comparing observations against predictions within same time range to calculate rmse
    pred_static = results.get_prediction(start=start_date, dynamic=False).predicted_mean
    pred_dynamic = results.get_prediction(start=start_date, dynamic=True).predicted_mean
    observed = query[feature]
    print('Static RMSE: ', np.mean((observed - pred_static) ** 2))
    print('Dynamic RMSE: ', np.mean((observed - pred_dynamic) ** 2))


    # Get forecast up to query_date in months
    days_between = (query_date - end_date).days

    # relativedelta code to convert days to months based on this answer on stackoverflow:
    # https://stackoverflow.com/questions/32083726/how-do-i-convert-days-into-years-and-months-in-python
    rdelta = relativedelta(query_date, end_date)
    stepz = abs(int((rdelta.years*12 + rdelta.months)))

    forecast = results.get_forecast(steps=stepz)
    forecast_ci = forecast.conf_int()

    plugging_hole = pd.DataFrame(forecast.predicted_mean.head(1)).rename(columns={'predicted_mean':feature})
    query = pd.concat([query, plugging_hole])

    ax = query.plot(label='observed', figsize=(20, 4), title='Forecast')
    forecast.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    ax.set_xlim([end_date - timedelta(days=5*365), query_date])
    ax.legend()

    # forecast results
    print('Forecast: ',round(forecast.predicted_mean[len(forecast.predicted_mean)-1], 2))
    print('End of dataset: ', round(query.tail(2)[feature][0],2))
    return round(forecast.predicted_mean[len(forecast.predicted_mean)-1], 2)

    # Short-term hourly forecaster. Requires ./datasets/auckland_classified_complete.csv
# Use this function for short term hourly forecasts, e.g., what will the temperature be tomorrow at 9am?
# I used some code from this tutorial: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

# Parameters:
# 'feature' is the dataset column you want to forecast for, e.g., you want to predict 'Temp(C)'
# 'dt' is the date and time you want to predict for, as a string in this format: '2022-07-10:0900'
# 'history_days' is how many days from the end of the dataset going back to train on.

def hourly_forecast(feature, dt, history_days):
    # merge date and time into one and set as index column
    query = pd.read_csv('./datasets/auckland_classified_complete.csv')
    query['datetime'] = pd.to_datetime(query['Date'].astype(str) + ' ' + query['Time'].astype(str))
    query = query.drop(['Date','Time'],axis=1)
    query = query.rename(columns={'datetime':'Date'})
    query = query.set_index('Date')

    # getting data prior to the forecast date; the amount of data is controlled by the window size parameter
    query = query[[feature]]
    query = query.resample('H').mean()
    query_date = datetime.strptime(dt, '%Y-%m-%d:%H%M')
    end_date = pd.to_datetime(query.tail(1).index[0])
    start_date = end_date - timedelta(days=history_days)  # Fix: Added "days=" before history_days
    if ((query_date - end_date).days < 0):
            print("\nQuery date is already in observed history. Pick a future date.")
            return

    filt = (query.index >= start_date) & (query.index <= end_date)
    query = query[filt]
    query.plot(figsize=(20, 4), title="History", ylabel=feature)

    # generate all different combinations of p, d and q triplets, find optimal combination (ie produces a model with lowest AIC)
    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, d, q))
    print('\nComputing', feature, end='')
    aics = {'param':[], 'score':[]}
    for param in pdq:
        try:
            mod = ARIMA(query[feature], order=param)
            results = mod.fit()
            print('.', end='')
            aics['param'].append(param)
            aics['score'].append(results.aic)
        except:
            continue

    aics = pd.DataFrame(aics)
    aics = aics[aics.score > 0]
    print('\n\nCombination with best AIC score:')
    print(aics[aics.score == min(aics.score)],'\n')
    param = aics[aics.score == min(aics.score)]['param'].iloc[0]

    # training model with optimal p d q combo
    mod = ARIMA(query[feature], order=param)
    results = mod.fit()
    print(results.summary().tables[1])

    # comparing observations against predictions within same time range to calculate rmse
    pred_static = results.get_prediction(start=start_date, dynamic=False).predicted_mean
    pred_dynamic = results.get_prediction(start=start_date, dynamic=True).predicted_mean
    observed = query[feature]
    print('Static RMSE: ', np.mean((observed - pred_static) ** 2))
    print('Dynamic RMSE: ', np.mean((observed - pred_dynamic) ** 2))

    # get forecast for however many hours ahead of the end of the dataset
    hours = abs(int((query_date - end_date).seconds / 60 / 60))
    days_hours = abs(int((query_date - end_date).days) * 24)


    # plots the forecast
    plugging_hole = pd.DataFrame(forecast.predicted_mean.head(1)).rename(columns={'predicted_mean':feature})
    query = pd.concat([query, plugging_hole])
    ax = query.plot(label='observed', figsize=(20, 4), title='Forecast')
    forecast.predicted_mean.plot(ax = ax, label='Forecast', figsize=(20, 3), title='Forecast')
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='k', alpha=.25)

    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    ax.set_xlim([end_date - timedelta(days=1), query_date])
    ax.legend()

    # forecast results
    print('Forecast: ', round(forecast.predicted_mean[len(forecast.predicted_mean)-1],2))
    print('End of set: ', round(query.tail(2)[feature][0],2))
    return round(forecast.predicted_mean[len(forecast.predicted_mean)-1],2)
    # Mid-term daily forecaster. Requires ./datasets/auckland_classified_complete.csv
# Use this function for daily forecasts, e.g., what will the mean temperature for tomorrow be?
# I used some code from this tutorial: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

# Parameters:
# 'feature' is the dataset column you want to forecast for, e.g., you want to predict 'Temp(C)'
# 'dt' is the date you want to predict for, as a string in this format: '2022-07-10'
# 'history_days' is how many days prior to day of prediction you want the model to train on, e.g., 30

def daily_forecast(feature, dt, history_days):
    # trimming down the classified set to dates from end to end - history_days
    query = pd.read_csv('./datasets/auckland_classified_complete.csv')
    query['datetime'] = pd.to_datetime(query['Date'].astype(str) + ' ' + query['Time'].astype(str))
    query = query.drop(['Date','Time'],axis=1)
    query = query.rename(columns={'datetime':'Date'})
    query = query.set_index('Date')
    query = query[[feature]]
    query = query.resample('D').mean()
    query_date = datetime.strptime(dt, '%Y-%m-%d')
    end_date = pd.to_datetime(query.tail(1).index[0])
    start_date = end_date - timedelta(history_days)

    if ((query_date - end_date).days < 0):
        print("\nQuery date is already in observed history. Pick a future date.")
        return

    filt = (query.index >= start_date) & (query.index <= end_date)
    query = query[filt]
    query.plot(figsize=(20, 4), title="History", ylabel=feature)

    # generate different combinations of p, d and q triplets, find optimal combination (ie produces a model with lowest AIC)
    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, d, q))
    print('\nComputing', feature, end='')
    aics = {'param':[], 'score':[]}
    for param in pdq:
        try:
            mod = ARIMA(query[feature], order=param)
            results = mod.fit()
            print('.', end='')
            aics['param'].append(param)
            aics['score'].append(results.aic)
        except:
            continue

    aics = pd.DataFrame(aics)
    aics = aics[aics.score > 0]
    print('\n\nCombination with best AIC score:')
    print(aics[aics.score == min(aics.score)], '\n')
    param = aics[aics.score == min(aics.score)]['param'].iloc[0]

    # training model with optimal p d q combo
    mod = ARIMA(query[feature], order=param)
    results = mod.fit()
    print(results.summary().tables[1])

    # comparing observations against predictions within same time range to calculate rmse
    pred_static = results.get_prediction(start=start_date, dynamic=False).predicted_mean
    pred_dynamic = results.get_prediction(start=start_date, dynamic=True).predicted_mean
    observed = query[feature]
    print('Static RMSE: ', np.mean((observed - pred_static) ** 2))
    print('Dynamic RMSE: ', np.mean((observed - pred_dynamic) ** 2))

    # get forecast for however many days ahead and plot
    stepz = (query_date - end_date).days
    if (stepz < 0):
        print("\nQuery date is already in observed history. Pick a future date.")
        return

    forecast = results.get_forecast(steps=stepz)
    forecast_ci = forecast.conf_int()

    # plots the forecast
    plugging_hole = pd.DataFrame(forecast.predicted_mean.head(1)).rename(columns={'predicted_mean':feature})
    query = pd.concat([query, plugging_hole])
    ax = query.plot(label='observed', figsize=(20, 4), title='Forecast')
    forecast.predicted_mean.plot(ax = ax, label='Forecast', figsize=(20, 3), title='Forecast')
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='k', alpha=.25)

    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    ax.set_xlim([end_date, query_date])
    ax.legend()

    print('Forecast: ',round(forecast.predicted_mean[len(forecast.predicted_mean)-1],2))
    print('End of set: ', round(query.tail(2)[feature][0],2))
    return round(forecast.predicted_mean[len(forecast.predicted_mean)-1],2)

    # Short-term condition forecaster. Requires ./datasets/auckland_classified_complete.csv
# Use this function for short term condition forecasts, e.g., what will the condition be tomorrow at 9am?
# Some of the K-Folds Cross Validation code here, also used in last assignment, was based on https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/

# Parameters:
# 'dt' is the date and time you want to predict for, as a string in this format: '2022-07-10:0900'
# 'history_days' is how many days from the end of the dataset going back to train on.

def hourly_forecast_condition(dt, history_days):
    temp = hourly_forecast('Temp(C)', dt, history_days)
    dew = hourly_forecast('DewPoint(C)', dt, history_days)
    humidity = hourly_forecast('Humidity(%)', dt, history_days)
    windspeed = hourly_forecast('WindSpeed(m/s)', dt, history_days)
    windgust = hourly_forecast('WindGust(m/s)', dt, history_days)
    pressure = hourly_forecast('Pressure(hPa)', dt, history_days)
    precipitation = hourly_forecast('Precipitation(mm)', dt, history_days)

    df_query = pd.read_csv('./datasets/auckland_classified_complete.csv')
    df_query = df_query.drop(['Date','Time','Wind'], axis = 1)
    df_query = df_query.convert_dtypes()
    df_query = df_query.dropna().reset_index(drop=True)

    X = df_query.drop('Condition',axis=1).to_numpy(dtype=float)
    y = df_query['Condition']
    X_scaler = preprocessing.RobustScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    # scaled model performance
    model_s = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    evaluator_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    evaluations_s = []

    for train_index, forecast_index in evaluator_s.split(X_scaled, y):
        X_train_fold, X_forecast_fold = X_scaled[train_index], X_scaled[forecast_index]
        y_train_fold, y_forecast_fold = y[train_index], y[forecast_index]
        model_s.fit(X_train_fold, y_train_fold)
        evaluations_s.append(model_s.score(X_forecast_fold, y_forecast_fold))

    print('\n-- scaled --')
    print('upper accuracy: ', np.max(evaluations_s)*100, '%')
    print('lower accuracy: ', np.min(evaluations_s)*100, '%')
    print('mean accuracy: ', np.mean(evaluations_s)*100, '%')
    print('SD: ', np.std(evaluations_s))

    forecast = {
        'Temp(C)':[temp],
        'DewPoint(C)':[dew],
        'Humidity(%)':[humidity],
        'WindSpeed(m/s)':[windspeed],
        'WindGust(m/s)':[windgust],
        'Pressure(hPa)':[pressure],
        'Precipitation(mm)':[precipitation]
    }

    forecast = pd.DataFrame(forecast)
    forecast = forecast.assign(Predicted_Scaled = lambda row: model_s.predict(
        row[['Temp(C)','DewPoint(C)','Humidity(%)','WindSpeed(m/s)','WindGust(m/s)','Pressure(hPa)','Precipitation(mm)']]))

    print('\nForecast: ',forecast['Predicted_Scaled'][0])
    return forecast['Predicted_Scaled'][0]




def prepareDataframe_combo_20(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Day(Local_Date)'] = pd.to_datetime(df['Day(Local_Date)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Day(Local_Date)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2003', after='2023')
    return df

def prepareDataframe_combo_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Day(Local_Date)'] = pd.to_datetime(df['Day(Local_Date)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Day(Local_Date)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df

def prepareDataframe_wind_20(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df['Date(NZST)'] = df['Date(NZST)'].dt.date #keeping just the date for the 20 year historical data, because only using daily periods
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)']) #this line exists because above line annoyingly converts from datetime to date, stopping me from using truncate
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2003', after='2023')
    return df

def prepareDataframe_rain_20(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df['Date(NZST)'] = df['Date(NZST)'].dt.date
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'])
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2003', after='2023')
    return df

def prepareDataframe_temps_20(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df['Date(NZST)'] = df['Date(NZST)'].dt.date
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'])
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2003', after='2023')
    return df

def prepareDataframe_wind_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df

def prepareDataframe_rain_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df

def prepareDataframe_humidity_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df

def prepareDataframe_temps_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df

def prepareDataframe_sunshine_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df = df.drop(columns='Freq')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df

def prepareDataframe_pressure_now(csvFile):
    df = pd.read_csv(csvFile)
    df = df.drop_duplicates()
    df['Date(NZST)'] = pd.to_datetime(df['Date(NZST)'], format='%Y%m%d:%H%M')
    df = df.rename(columns={'Date(NZST)':'Date'}).set_index('Date')
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].replace('-',np.nan)
    df.loc[:, df.columns != 'Station'] = df.loc[:, df.columns != 'Station'].astype('float64')
    df = df.convert_dtypes()
    df = df.truncate(before='2022')
    return df


auckland = {}
wellington = {}
christchurch = {}

auckland['combo_20'] = prepareDataframe_combo_20('./datasets/auckland_combo_2003-2023.csv')
auckland['wind_20'] = prepareDataframe_wind_20('./datasets/auckland_wind_2003-2023.csv') # 9am daily, pukekohe station
auckland['rain_20'] = prepareDataframe_rain_20('./datasets/auckland_rain_2003-2023.csv')
auckland['temps_20'] = prepareDataframe_temps_20('./datasets/auckland_temps_2003-2023.csv')

auckland['combo_now'] = prepareDataframe_combo_now('./datasets/auckland_combo_2022-now.csv')
auckland['wind_now'] = prepareDataframe_wind_now('./datasets/auckland_wind_2022-now.csv') # hourly surface wind, measured at sky tower
auckland['rain_now'] = prepareDataframe_rain_now('./datasets/auckland_rain_2022-now.csv') # hourly rain table data, measured in albany north shore
auckland['humidity_now'] = prepareDataframe_humidity_now('./datasets/auckland_humidity_2022-now.csv') # hourly humidity, measured in albany north shore
auckland['temps_now'] = prepareDataframe_temps_now('./datasets/auckland_temps_2022-now.csv') # hourly temperatures, measured in albany north shore
auckland['sunshine_now'] = prepareDataframe_sunshine_now('./datasets/auckland_sunshine_2022-now.csv') # daily sunshine in hours, measured in albany north shore
auckland['pressure_now'] = prepareDataframe_pressure_now('./datasets/auckland_pressure_2022-now.csv') # hourly pressure, measured in albany north shore



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime as dt

# Load the weather data
@st.cache_data
def load_data():
    return auckland['combo_20']

data = load_data()
# Define a threshold for missing values
threshold = 0.3

# Find the columns where the percentage of missing values is greater than the threshold
cols_to_drop = data.columns[data.isnull().mean() > threshold]

# Drop those columns
data = data.drop(cols_to_drop, axis=1)

# Then drop rows with missing values
data = data.dropna()

# Show a quick overview of the data
st.write(data.head())

# Add a header
st.title("Weather Forecast in New Zealand")

# Let user select a station
station = st.selectbox('Select a station', data['Station'].unique())

# Filter data based on selected station
station_data = data[data['Station'] == station]
# Drop NaN values
station_data = station_data.dropna()
print(station_data)
# Display the data as a table for the selected station
st.write(station_data)

# Transform the index (which is 'Date') into a form that can be used by the model
dates_ordinal = station_data.index.map(dt.datetime.toordinal).values.reshape(-1,1)

# Split the dataset into 'features' and 'target'
#X = dates_ordinal
#y = station_data['Tdry(C)'].values.reshape(-1,1)
# create a list of numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# let user select target variable
target = st.sidebar.selectbox('Select a numeric column to predict', numeric_columns)

# define features and target
X = station_data.drop(target, axis=1)
X = X.select_dtypes(include=['int64', 'float64'])  # selecting only numeric columns
y = station_data[target]

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print(X_train)
# Train the model
#model = LinearRegression()
#model.fit(X_train, y_train)

# Prediction for the next 7 days
#next_week = pd.date_range(start=station_data.index.max(), periods=8)[1:]
#next_week_ordinal = [i.toordinal() for i in next_week]

#predictions = model.predict(np.array(next_week_ordinal).reshape(-1, 1))

# Print predictions for the next week
#st.write(pd.DataFrame({'Date': next_week, 'Predicted Dry Bulb Temperature (C)': predictions.flatten()}))
#
# let user select type of plot and column to plot
plot_type = st.sidebar.selectbox('Select a plot type', ['Bar', 'Line'])
column = st.sidebar.selectbox('Select a column to plot', data.columns)

# generate plot
if plot_type == 'Bar':
    fig = px.bar(data, x=data.index, y=column)
elif plot_type == 'Line':
    fig = px.line(data, x=data.index, y=column)

# display plot
st.plotly_chart(fig)

# let user select algorithm
algorithm = st.sidebar.selectbox(
    'Select an algorithm',
    ('Linear Regression', 'Random Forest', 'SVR'))

# create and train model
if algorithm == 'Linear Regression':
    model = LinearRegression()
elif algorithm == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, random_state=0)
else:
    model = SVR(kernel='linear', C=1.0)

model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)
last_known_values = X_test[-10:]

# Predict next 10 steps
predictions = model.predict(last_known_values)
# Display predictions
st.title('Weather Forecasting')
st.header('Predictions for the Next 10 Steps')
st.subheader('These are the predicted values for the selected location:')
# Display predictions
st.write(predictions)

# calculate and display mse
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error of {algorithm} on test set: {mse}')
