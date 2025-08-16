# stock-prediction-experiments-arima-lstm-xgboost

## Stock Price Trend Prediction: An R&D Approach Using ARIMA, LSTM & XGBoost

A research-driven project focused on predicting Apple Inc. (AAPL) stock prices using a combination of classical time series models and advanced machine learning techniques. This study compares the performance of ARIMA, Linear Regression, LSTM, and a hybrid LSTM-XGBoost model in next-day stock price forecasting.

## 📌 Project Overview

- **Domain:** Financial Time Series Forecasting  
- **Goal:** Predict next-day closing price of AAPL stock  
- **Models Compared:** ARIMA, Linear Regression, LSTM, LSTM-XGBoost Hybrid  
- **Dataset:** Yahoo Finance (1980–2025)  
- **Evaluation Metrics:** MSE, RMSE, MAPE  
- **Outcome:** Traditional models (ARIMA, Linear Regression) outperformed deep learning approaches in this case.

---

## 📌 Table of Contents

- [Executive Summary](#executive-summary)
- [Key Findings](#key-findings)
- [Modeling Approach](#modeling-approach)
- [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Future Work](#future-work)
- [Authors & Acknowledgements](#authors--acknowledgements)

---

## 📌 Executive Summary

Stock market forecasting is notoriously difficult due to high volatility, noise, and external influencing factors. In this project, we explored multiple modeling approaches to predict the next day's closing price of Apple stock using historical data. The models were evaluated on how well they capture temporal dependencies, trends, and non-linear patterns.

---

## 📌 Key Findings

> **Insight:** Despite the hype around deep learning, simpler models like ARIMA and Linear Regression performed better for short-term forecasting of stock prices.

---

## 📌 Modeling Approach

- **ARIMA:** Captures trend and seasonality with differencing and ACF/PACF analysis.
- **Linear Regression:** Baseline using lag features (Lag_1, Lag_2, Lag_3).
- **LSTM:** Sequence of past 10 days of closing prices as input.
- **LSTM + XGBoost:** Hybrid model combining sequential memory with gradient boosting for non-linear patterns.

---

## 📌 Data Preparation & Feature Engineering

- **Source:** Yahoo Finance (via `yfinance` library)
- **Period:** 1980-12-12 to 2025-02-28
- **Features Used:**
  - `Close`, `Open`, `High`, `Low`, `Volume`
  - Lag Features (`Lag_1`, `Lag_2`, `Lag_3`)
  - Technical Indicators (in future work)
- **Transformations:**
  - Differencing for ARIMA
  - MinMax/Standard Scaling for ML & LSTM models

---

## 📌 Technologies Used

- **Languages:** Python
- **Libraries & Tools:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `statsmodels`
  - `tensorflow/keras`, `xgboost`
  - `yfinance`, `Google Colab`, `Jupyter Lab`

---



