import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
#import the csv file
df = pd.read_csv('city_day.csv', parse_dates=['Date']).set_index('Date')
cols = [c for c in ['AQI', 'PM2.5', 'NO2', 'O3', 'temp'] if c in df.columns]
df = df[cols].dropna()
df['AQI_lag1'] = df['AQI'].shift(1)
df.dropna(inplace=True)
#Anlayze the frame data in csv

scaler = MinMaxScaler()
features_to_scale = [c for c in ['PM2.5', 'NO2', 'O3', 'AQI_lag1'] if c in df.columns]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
features = [c for c in ['PM2.5', 'NO2', 'O3', 'AQI_lag1'] if c in df.columns]
X = df[features]
y = df['AQI']
#test the train variable of X,y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  
)
#using RandomForest Method
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(" Random Forest Evaluation:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")

#plot show prediction 
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_rf, label='Predicted RF')
plt.title('Random Forest Regression')
plt.legend()
plt.show()
try:#any error occur using try catch method in the test
    arima_model = ARIMA(y_train, order=(5,1,0)).fit()
    forecast_arima = arima_model.forecast(steps=len(y_test))
except:
    forecast_arima = [y_test.mean()] * len(y_test)

rmse_arima = np.sqrt(mean_squared_error(y_test, forecast_arima))
print(f" ARIMA RMSE: {rmse_arima:.2f}")

# Plot ARIMA results
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, forecast_arima, label='Predicted ARIMA')
plt.title('ARIMA Forecasting')
plt.legend()
plt.show()
hybrid_pred = (np.array(y_pred_rf) + np.array(forecast_arima)) / 2
rmse_hybrid = np.sqrt(mean_squared_error(y_test, hybrid_pred))

print(f" Hybrid RMSE: {rmse_hybrid:.2f}")
print(f" Best RMSE: {min(rmse_rf, rmse_arima, rmse_hybrid):.2f}")
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, hybrid_pred, label='Predicted Hybrid')
plt.title('Hybrid Model (RF + ARIMA)')
plt.legend()
plt.show()


