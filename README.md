# 🌧️ Australian Rain Prediction — Streamlit App

A web application that predicts whether it will rain tomorrow at a given
location in Australia, based on today's weather conditions. Built with
**scikit-learn** and deployed as an interactive **Streamlit** app.

**🔗 Live demo:** (https://a34b9e6ce090a71c858877b1dd9d66375dc8c2c6.streamlit.app)

---

## Overview

This project takes the classic "Rain in Australia" weather dataset and turns
a trained machine learning model into a usable, interactive tool. Instead of
just presenting accuracy metrics in a notebook, the app lets a user pick a
location, adjust weather sliders (temperature, rainfall, wind, humidity), and
get an instant prediction with a probability score — a full path from model
to product.

## Features

- **Interactive input form** — sliders and dropdowns for temperature,
  rainfall, wind direction/speed, humidity, and location, built with native
  Streamlit widgets.
- **Real-time inference** — the trained pipeline (imputer → scaler → one-hot
  encoder → logistic regression) runs directly in the app to return a
  `Rain` / `No Rain` prediction with a probability score.
- **Exploratory data view** — a table of average weather statistics grouped
  by location, giving users context before making a prediction.
- **Reproducible pipeline** — preprocessing steps are stored inside the same
  `.joblib` artifact as the model, so inference always matches training-time
  transformations exactly.

## Tech Stack

| Layer          | Tool                                   |
|----------------|-----------------------------------------|
| Language       | Python 3                                |
| ML / Data      | scikit-learn, pandas, NumPy             |
| Model          | Logistic Regression                     |
| Web framework  | Streamlit                               |
| Model storage  | joblib                                  |
| Deployment     | Streamlit Community Cloud               |

## Project Structure

```
├── rain_app.py           # Streamlit application
├── requirements.txt       # Python dependencies
├── models/
│   └── aussie_rain.joblib # trained model + preprocessing pipeline
├── data/
│   └── weatherAUS.csv     # training dataset (Kaggle: Rain in Australia)
└── images/
    └── rain.png           # header image
```

## How It Works

1. The user selects a location and adjusts weather inputs via sliders and
   dropdowns.
2. On submit, the input is assembled into a single-row DataFrame matching the
   model's expected schema.
3. The same preprocessing pipeline used during training — missing value
   imputation, feature scaling, and one-hot encoding — is applied to the
   input.
4. The trained logistic regression model outputs a class prediction
   (`Yes`/`No`) along with the associated probability, which is displayed to
   the user.


## Dataset

Trained on the [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
dataset from Kaggle, which contains ~10 years of daily weather observations
from multiple Australian weather stations.

## Possible Improvements

- Add model performance metrics (ROC-AUC, confusion matrix) to the app.
- Experiment with additional models (Random Forest, XGBoost) for comparison.
- Add unit tests for the preprocessing pipeline.
- Cache the dataset load with `st.cache_data` for faster reloads.

---

*This project was built as a hands-on exercise in taking a trained ML model
from a notebook to a deployed, user-facing application.*
