# Gold Price Forecasting & Macroeconomic Indicator Modeling
### Data Science and Advanced Programming — Project
### Kevin Murengezi — HEC Lausanne (MSch)

---

## 📌 Project Overview
The aim of this project is to build a machine learning–based system to analyze how macroeconomic variables influence gold prices and to generate short-term forecasts.

This project combines:
- Real financial data (gold prices, interest rates, USD index, central bank demand)
- Feature engineering based on economic theory
- Predictive modeling (Linear Regression + ARIMA)
- Model evaluation
- Interpretation of results

---

## 🎯 Objectives

### **Main Goals**
1. Clean and preprocess gold-related and macroeconomic time-series data  
2. Engineer features representing fundamental drivers of gold  
   - Fed nominal interest rates  
   - USD index  
   - Central bank gold demand  
   - ETF gold flows  
3. Build a **Linear Regression model**  
4. Build an **ARIMA model**  
5. Compare both models  
6. Provide an economic interpretation of model predictions  

### **Stretch Goals (optional)**
- Hyperparameter tuning  
- Grid search on ARIMA  
- Lasso/Ridge regression  

---

## 📁 Repository Structure

```plaintext
project/
│
├── data/
│   ├── gold_prices.csv
│   ├── fed_rates.csv
│   ├── usd_index.csv
│   └── central_bank_demand.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_linear_regression_model.ipynb
│   ├── 04_arima_model.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
│
├── README.md
└── requirements.txt
