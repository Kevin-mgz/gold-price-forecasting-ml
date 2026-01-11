# Gold Price Forecasting with Machine Learning

A machine learning system for predicting weekly gold price movements using Random Forest classification.

## 📁 Project Structure

```
gold-price-forecasting-ml/
├── data/
│   ├── raw/                    # Downloaded CSV files
│   └── processed/              # Engineered features
├── models/                     # Trained model (.joblib)
├── results/                    # Plots, metrics, reports
├── src/
│   ├── data_loader.py          # Step 1: Download data
│   ├── feature_engineering.py  # Step 2: Create features
│   ├── models.py               # Step 3: Train model
│   ├── evaluation.py           # Step 4: Backtest
│   ├── predict.py              # CLI prediction
│   └── dashboard.py            # Streamlit web app
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Run Full Pipeline

```bash
python src/data_loader.py         # Download data
python src/feature_engineering.py # Create features
python src/models.py              # Train model
python src/evaluation.py          # Evaluate & backtest
```

### Generate Prediction

```bash
python src/predict.py             # CLI prediction for next week
```

### Launch Dashboard

```bash
streamlit run src/dashboard.py    # Web interface at localhost:8501
```

## 📊 Results

| Metric | ML Strategy | Buy & Hold |
|--------|-------------|------------|
| Accuracy | 54.2% | - |
| Total Return | +96.39% | +127.25% |
| Sharpe Ratio | 1.36 | 1.40 |
| Max Drawdown | -13.16% | -17.79% |
| Win Rate | 61.05% | - |

## 👤 Author

Kevin Murengezi — University of Geneva — January 2026
