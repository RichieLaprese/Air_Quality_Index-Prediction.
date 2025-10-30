import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

#Streamlit UI

st.set_page_config(page_title="Air Quality Prediction",layout="wide")
st.title("🌤️ Air Quality Prediction using Random Forest, ARIMA, and Hybrid Model")
uploaded_file = st.file_uploader("📂 Upload your dataset (city_day.csv)", type=["csv"])
#import the csv file
if uploaded_file:
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
    
    st.subheader(" Random Forest Evaluation:")
    st.write(f"RMSE: {rmse_rf:.2f}")
    st.write(f"R² Score: {r2_rf:.2f}")
    
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(y_test.index, y_test, label='Actual')
    ax1.plot(y_test.index, y_pred_rf, label='Predicted RF')
    ax1.set_title('Random Forest Regression')
    ax1.legend()
    st.pyplot(fig1)
    

#any error occur using try catch method in the test
    try:
        arima_model = ARIMA(y_train, order=(5,1,0)).fit()
        forecast_arima = arima_model.forecast(steps=len(y_test))
    except:
        forecast_arima = [y_test.mean()] * len(y_test)

    rmse_arima = np.sqrt(mean_squared_error(y_test, forecast_arima))
    st.subheader("📈 ARIMA Results")
    st.write(f"**RMSE:** {rmse_arima:.2f}")


# Plot ARIMA results
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(y_test.index, y_test, label='Actual')
    ax2.plot(y_test.index, forecast_arima, label='Predicted ARIMA')
    ax2.set_title('ARIMA Forecasting')
    ax2.legend()
    st.pyplot(fig2)
    
    
    hybrid_pred = (np.array(y_pred_rf) + np.array(forecast_arima)) / 2
    rmse_hybrid = np.sqrt(mean_squared_error(y_test, hybrid_pred))


    st.subheader("🤖 Hybrid Model Results")
    st.write(f"**Hybrid RMSE:** {rmse_hybrid:.2f}")
    st.write(f"🏆 **Best RMSE:** {min(rmse_rf, rmse_arima, rmse_hybrid):.2f}")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(y_test.index, y_test, label='Actual')
    ax3.plot(y_test.index, hybrid_pred, label='Predicted Hybrid')
    ax3.set_title('Hybrid Model (RF + ARIMA)')
    ax3.legend()
    st.pyplot(fig3)
else:
    st.warning("👆 Please upload a`city_day.csv`file to get start.")

