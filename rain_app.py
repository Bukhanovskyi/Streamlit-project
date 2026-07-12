import streamlit as st
import pandas as pd
import numpy as np
import joblib


def predict(input_data: dict):
    artifacts = joblib.load('models/aussie_rain.joblib')

    model = artifacts['model']
    imputer = artifacts['imputer']
    scaler = artifacts['scaler']
    encoder = artifacts['encoder']
    input_cols = artifacts['input_cols']
    numeric_cols = artifacts['numeric_cols']
    categorical_cols = artifacts['categorical_cols']
    encoded_cols = artifacts['encoded_cols']

    df = pd.DataFrame([input_data], columns=input_cols)
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df[encoded_cols] = encoder.transform(df[categorical_cols])

    X = df[numeric_cols + encoded_cols]
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][list(model.classes_).index('Yes')]
    return prediction, probability


# Заголовок застосунку
st.title('Прогноз дощу в Австралії')
st.markdown('Це проста модель (логістична регресія) для прогнозування, \
чи піде дощ завтра, на основі сьогоднішніх погодних показників.')
st.image('images/rain.png')

# Відображення таблиці середніх значень
st.header("Середні значення показників по містах")
weather_df = pd.read_csv("data/weatherAUS.csv")
mean_values = weather_df.groupby('Location').mean(numeric_only=True).reset_index()
st.dataframe(mean_values)

# Заголовок секції з погодними показниками
st.header("Погодні показники")
col1, col2 = st.columns(2)

# Введення температури та опадів
with col1:
    st.text("Температура та опади")
    min_temp = st.slider('MinTemp (°C)', -10.0, 35.0, 13.0)
    max_temp = st.slider('MaxTemp (°C)', 0.0, 50.0, 23.0)
    temp9am = st.slider('Temp9am (°C)', -5.0, 40.0, 17.0)
    temp3pm = st.slider('Temp3pm (°C)', -5.0, 45.0, 21.0)
    rainfall = st.slider('Rainfall (mm)', 0.0, 100.0, 0.0)
    rain_today = st.selectbox('RainToday', ['No', 'Yes'])

# Введення вітру, вологості та тиску
with col2:
    st.text("Вітер, вологість, тиск")
    location = st.selectbox('Location', sorted(weather_df['Location'].unique()))
    wind_gust_dir = st.selectbox('WindGustDir', sorted(weather_df['WindGustDir'].dropna().unique()))
    wind_dir_9am = st.selectbox('WindDir9am', sorted(weather_df['WindDir9am'].dropna().unique()))
    wind_dir_3pm = st.selectbox('WindDir3pm', sorted(weather_df['WindDir3pm'].dropna().unique()))
    wind_gust_speed = st.slider('WindGustSpeed (km/h)', 0.0, 130.0, 40.0)
    humidity_9am = st.slider('Humidity9am (%)', 0.0, 100.0, 70.0)
    humidity_3pm = st.slider('Humidity3pm (%)', 0.0, 100.0, 50.0)

# Кнопка для прогнозування
if st.button("Прогнозувати дощ"):
    input_data = {
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': weather_df['Evaporation'].mean(),
        'Sunshine': weather_df['Sunshine'].mean(),
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': weather_df['WindSpeed9am'].mean(),
        'WindSpeed3pm': weather_df['WindSpeed3pm'].mean(),
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': weather_df['Pressure9am'].mean(),
        'Pressure3pm': weather_df['Pressure3pm'].mean(),
        'Cloud9am': weather_df['Cloud9am'].mean(),
        'Cloud3pm': weather_df['Cloud3pm'].mean(),
        'Temp9am': temp9am,
        'Temp3pm': temp3pm,
        'RainToday': rain_today,
    }

    # Викликаємо функцію прогнозування
    result, probability = predict(input_data)
    st.write(f"Прогноз: {'Дощ буде' if result == 'Yes' else 'Дощу не буде'} "
             f"(ймовірність дощу: {probability * 100:.1f}%)")