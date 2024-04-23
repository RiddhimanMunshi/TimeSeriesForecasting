#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("AirPassengers (1).csv")
df.head(5)

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")


# In[2]:


df.info()


# In[3]:


# Convert 'month' column to datetime format
df['Month'] = pd.to_datetime(df['Month'])

# Set 'month' column as the index
df.set_index('Month', inplace=True)


# In[4]:


# Plot the time series data
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['#Passengers'], color='blue')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[5]:


from statsmodels.tsa.seasonal import seasonal_decompose
# Perform seasonal decomposition
result = seasonal_decompose(df['#Passengers'], model='additive')

# Plot the decomposed components
result.plot()


# In[6]:


from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(df['#Passengers'])

# Extract and print the results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')


# In[7]:


df['Passengers_diff'] = df['#Passengers'].diff(1)  # Take the first difference

# Drop NaN values resulting from differencing
df.dropna(inplace=True)

# Perform ADF test
result = adfuller(df['Passengers_diff'])

# Extract and print the results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')


# In[8]:


df['Passengers_diff'] = df['#Passengers'].diff(2)  # Take the second difference

# Drop NaN values resulting from differencing
df.dropna(inplace=True)

# Perform ADF test
result = adfuller(df['Passengers_diff'])

# Extract and print the results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')


# In[9]:


# Plot the time series data
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['Passengers_diff'], color='blue')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[10]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Plot ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot
plot_acf(df['Passengers_diff'], ax=ax1, lags=12)
ax1.set_title('Autocorrelation Function (ACF)')

# PACF plot
plot_pacf(df['Passengers_diff'], ax=ax2, lags=12)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


# In[18]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from itertools import product


# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Define the hyperparameter grid
p_values = [0, 1, 2]  # AR order
d_values = [0, 1]  # Differencing order
q_values = [0, 1, 2]  # MA order
P_values = [0, 1, 2]  # Seasonal AR order
D_values = [0, 1]  # Seasonal differencing order
Q_values = [0, 1, 2]  # Seasonal MA order
s_values = [12]  # Seasonal period

# Create a grid of hyperparameters combinations
param_grid = product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

# Perform grid search
best_score = np.inf
best_params = None

for params in param_grid:
    try:
        # Fit SARIMAX model
        model = SARIMAX(train_data['Passengers_diff'], order=params[:3], seasonal_order=params[3:7])
        results = model.fit()

        # Forecast
        forecast = results.forecast(steps=len(test_data))

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data['Passengers_diff'], forecast))

        # Update best parameters if RMSE improves
        if rmse < best_score:
            best_score = rmse
            best_params = params
    except:
        continue

# Print the best parameters
print("Best Parameters:", best_params)
print("Best RMSE:", best_score)


# In[12]:


pip install pmdarima


# In[14]:


import pmdarima as pm

# Fit the ARIMA model using auto_arima
model = pm.auto_arima(train_data['Passengers_diff'], seasonal=True, m=12, trace=True, suppress_warnings=True)

# Forecast future values
n_periods = len(test_data)
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# Plot actual vs forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Passengers_diff'], label='Actual')
plt.plot(test_data.index, forecast, label='Forecast', color='red')
plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.title('Actual vs Forecasted Passengers')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()


# In[26]:


model = SARIMAX(train_data['Passengers_diff'], order=(0, 0, 1), seasonal_order=(0, 1, 0, 12))
results = model.fit()

# Forecast future values on the testing data
forecast = results.forecast(steps=len(test_data))  # Forecasting on testing data

# Plot actual vs predicted values on the testing data
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Passengers_diff'], label='Actual')
plt.plot(test_data.index, forecast, label='Predicted', color='red')
plt.title('Actual vs Predicted Passengers (Testing Data)')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()


# In[32]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test_data['Passengers_diff'], forecast))
mape = np.mean(np.abs((test_data['Passengers_diff'] - forecast) / test_data['Passengers_diff'])) * 100

# Print model summary, RMSE, and MAPE
print("Model Summary:")
print(results.summary())
print("\nRoot Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)


# In[28]:


#HYPER PARAMETER TUNING FOR BEST MAPE SCORE


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Define the hyperparameter grid
p_values = [0, 1, 2]  # AR order
d_values = [0, 1]  # Differencing order
q_values = [0, 1, 2]  # MA order
P_values = [0, 1, 2]  # Seasonal AR order
D_values = [0, 1]  # Seasonal differencing order
Q_values = [0, 1, 2]  # Seasonal MA order
s_values = [12]  # Seasonal period

# Create a grid of hyperparameters combinations
param_grid = product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

# Perform grid search
best_score = np.inf
best_params = None

for params in param_grid:
    try:
        # Fit SARIMAX model
        model = SARIMAX(train_data['Passengers_diff'], order=params[:3], seasonal_order=params[3:7])
        results = model.fit()

        # Forecast
        forecast = results.forecast(steps=len(test_data))

        # Calculate MAPE
        mape = mean_absolute_percentage_error(test_data['Passengers_diff'], forecast)

        # Update best parameters if MAPE improves
        if mape < best_score:
            best_score = mape
            best_params = params
    except:
        continue

# Print the best parameters
print("Best Parameters:", best_params)
print("Best MAPE:", best_score)


# In[29]:


# Invert the differencing to obtain forecasted values in the original scale
forecast = np.cumsum(forecast_diff) + train_data['#Passengers'].iloc[-1]

# Print actual vs forecasted values
print("Actual Passengers:", test_data['#Passengers'].values)
print("Forecasted Passengers:", forecast)


# In[30]:


# Plot actual vs forecasted values
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['#Passengers'], label='Actual')
plt.plot(test_data.index, forecast, label='Forecasted', color='red')
plt.title('Actual vs Forecasted Passengers')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()


# In[ ]:




