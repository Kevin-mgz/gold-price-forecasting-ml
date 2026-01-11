# Gold Price Forecasting with Machine Learning

A machine learning system for predicting weekly gold price movements using Random Forest classification.

## 📁 Project Structure

```
gold-price-forecasting-ml/
├── data/
│   ├── raw/                    # Downloaded CSV files
│   │   └── central_bank_demand.csv  # Bundled dataset
│   └── processed/              # Generated: dataset_final.csv
├── results/                    # Generated: plots, metrics, reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Step 1: Download data
│   ├── clean_csv.py            # Clean central bank CSV
│   ├── feature_engineering.py  # Step 2: Create features
│   ├── models.py               # Step 3: Train classifier
│   ├── evaluation.py           # Backtest & metrics
│   ├── regression_bonus.py     # Step 4: Train regressor
│   ├── predict.py              # Step 5: Generate prediction
│   ├── dashboard.py            # Streamlit web app
│   └── debug_data.py           # Data leakage diagnostic
├── main.py                     # Main entry point
├── project_report.md           # Academic report
├── project_report.pdf          # PDF version
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Setup

```bash
# Install dependencies (use pip3 on macOS, pip on Windows)
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run full pipeline (data → features → training → regression → prediction)
python main.py

# At the end, you will be prompted to launch the dashboard
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

Kevin Murengezi — University of Lausanne — January 2026
