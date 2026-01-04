"""
Gold Price Forecasting - Evaluation & Backtesting Script
Evaluates model performance and simulates trading strategy
Author: Kevin Murengezi
Date: 2025
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def setup_paths():
    """
    Setup project paths for model, data, and results.

    Returns:
        tuple: (model_path, data_path, results_dir)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    model_path = project_root / "models" / "random_forest_model.joblib"
    data_path = project_root / "data" / "processed" / "dataset_final.csv"
    results_dir = project_root / "results"

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    return model_path, data_path, results_dir


def load_model(model_path):
    """
    Load the trained Random Forest model.

    Args:
        model_path (Path): Path to the saved model

    Returns:
        model: Loaded scikit-learn model
    """
    print(f"📦 Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully")

    return model


def load_and_prepare_data(data_path):
    """
    Load dataset and separate features from target.
    Keep Gold_Close for financial analysis.

    Args:
        data_path (Path): Path to the processed dataset

    Returns:
        tuple: (X, y, gold_returns, dates)
    """
    print(f"\n📂 Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load with date as index
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Define columns to exclude from features
    reference_columns = ["Gold_Close", "DXY_Close", "Rates_10Y", "Real_Rates_10Y"]
    target_column = "Target"

    # Check if target exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Get feature columns (only numeric, excluding reference and target)
    exclude_columns = reference_columns + [target_column]
    potential_features = [col for col in df.columns if col not in exclude_columns]

    # Filter numeric features only
    numeric_features = [
        col for col in potential_features if pd.api.types.is_numeric_dtype(df[col])
    ]

    # Prepare data
    X = df[numeric_features].copy()
    y = df[target_column].copy()

    # Calculate gold returns for backtesting
    # If Gold_Return exists, use it; otherwise calculate from Gold_Close
    if "Gold_Return" in df.columns:
        gold_returns = df["Gold_Return"].copy()
    elif "Gold_Close" in df.columns:
        gold_returns = df["Gold_Close"].pct_change()
    else:
        raise ValueError("Neither 'Gold_Return' nor 'Gold_Close' found for backtesting")

    dates = df.index

    print(f"✓ Features: {len(numeric_features)} columns")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")

    return X, y, gold_returns, dates


def create_temporal_split(X, y, gold_returns, dates, test_size=0.2):
    """
    Split data temporally (last 20% for testing).

    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
        gold_returns (Series): Gold returns for backtesting
        dates (Index): Date index
        test_size (float): Proportion for test set

    Returns:
        tuple: (X_train, X_test, y_train, y_test, returns_test, dates_test)
    """
    print(f"\n📊 Creating temporal split ({int(test_size*100)}% test set)...")

    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))

    # Split data chronologically
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    returns_test = gold_returns.iloc[split_idx:]
    dates_test = dates[split_idx:]

    print(
        f"✓ Train set: {len(X_train)} samples ({dates[0].date()} to {dates[split_idx-1].date()})"
    )
    print(
        f"✓ Test set:  {len(X_test)} samples ({dates[split_idx].date()} to {dates[-1].date()})"
    )

    return X_train, X_test, y_train, y_test, returns_test, dates_test


def save_metrics_to_json(metrics, results_dir):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): Metrics to save
        results_dir (Path): Directory to save the JSON file
    """
    results_path = results_dir / "latest_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Metrics automatically saved to: {results_path}")
    print("=" * 70)


def evaluate_classification_metrics(model, X_test, y_test, results_dir):
    """
    Evaluate model performance with classification metrics.

    Args:
        model: Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        results_dir (Path): Directory to save metrics

    Returns:
        tuple: (y_pred, metrics_dict)
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION PERFORMANCE")
    print("=" * 70)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Display metrics
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Detailed classification report
    print("\n" + "-" * 70)
    print("Detailed Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=["Down (0)", "Up (1)"]))

    # Save metrics to JSON
    save_metrics_to_json(metrics, results_dir)

    return y_pred, metrics


def plot_confusion_matrix(y_test, y_pred, results_dir):
    """
    Create and save confusion matrix heatmap.

    Args:
        y_test (Series): True labels
        y_pred (array): Predicted labels
        results_dir (Path): Directory to save plot

    Returns:
        Path: Path to saved plot
    """
    print("\n📊 Generating confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Count"},
        xticklabels=["Predicted Down (0)", "Predicted Up (1)"],
        yticklabels=["Actual Down (0)", "Actual Up (1)"],
    )

    plt.title(
        "Confusion Matrix - Gold Price Direction Prediction",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")

    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(
        0.5,
        -0.15,
        f"Overall Accuracy: {accuracy:.2%}",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save plot
    plot_path = results_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Confusion matrix saved to: {plot_path}")

    return plot_path


def backtest_strategy(y_test, y_pred, returns_test, dates_test, results_dir):
    """
    Backtest trading strategy and compare with buy-and-hold.

    Strategy:
    - Predict 1 (Up): Go long (take the return)
    - Predict 0 (Down): Stay in cash (0% return) - Long-Only strategy

    Args:
        y_test (Series): True labels
        y_pred (array): Predicted labels
        returns_test (Series): Gold returns
        dates_test (Index): Dates
        results_dir (Path): Directory to save results

    Returns:
        dict: Backtesting results
    """
    print("\n" + "=" * 70)
    print("BACKTESTING - TRADING STRATEGY SIMULATION")
    print("=" * 70)

    # Clean data: remove NaN values
    valid_idx = ~returns_test.isna()
    returns_clean = returns_test[valid_idx].values
    y_pred_clean = y_pred[valid_idx]
    y_test_clean = y_test[valid_idx].values
    dates_clean = dates_test[valid_idx]

    # Strategy returns: only take position when predicting UP (1)
    strategy_returns = np.where(y_pred_clean == 1, returns_clean, 0)

    # Buy & Hold returns: always invested
    buyhold_returns = returns_clean

    # Calculate cumulative returns (starting with $1000)
    initial_capital = 1000

    strategy_equity = initial_capital * (1 + strategy_returns).cumprod()
    buyhold_equity = initial_capital * (1 + buyhold_returns).cumprod()

    # Calculate performance metrics
    strategy_total_return = (strategy_equity[-1] / initial_capital - 1) * 100
    buyhold_total_return = (buyhold_equity[-1] / initial_capital - 1) * 100

    strategy_sharpe = (
        np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        if np.std(strategy_returns) > 0
        else 0
    )
    buyhold_sharpe = (
        np.mean(buyhold_returns) / np.std(buyhold_returns) * np.sqrt(252)
        if np.std(buyhold_returns) > 0
        else 0
    )

    # Maximum drawdown
    strategy_cummax = np.maximum.accumulate(strategy_equity)
    strategy_drawdown = (strategy_equity - strategy_cummax) / strategy_cummax
    strategy_max_dd = strategy_drawdown.min() * 100

    buyhold_cummax = np.maximum.accumulate(buyhold_equity)
    buyhold_drawdown = (buyhold_equity - buyhold_cummax) / buyhold_cummax
    buyhold_max_dd = buyhold_drawdown.min() * 100

    # Win rate
    winning_trades = np.sum((y_pred_clean == 1) & (returns_clean > 0))
    total_trades = np.sum(y_pred_clean == 1)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Display results
    print(f"\n📈 STRATEGY PERFORMANCE (ML-Based Long-Only)")
    print("-" * 70)
    print(f"Total Return:      {strategy_total_return:>10.2f}%")
    print(f"Final Value:       ${strategy_equity[-1]:>10.2f}")
    print(f"Sharpe Ratio:      {strategy_sharpe:>10.2f}")
    print(f"Max Drawdown:      {strategy_max_dd:>10.2f}%")
    print(
        f"Win Rate:          {win_rate:>10.2f}% ({winning_trades}/{total_trades} trades)"
    )

    print(f"\n📊 BUY & HOLD PERFORMANCE (Benchmark)")
    print("-" * 70)
    print(f"Total Return:      {buyhold_total_return:>10.2f}%")
    print(f"Final Value:       ${buyhold_equity[-1]:>10.2f}")
    print(f"Sharpe Ratio:      {buyhold_sharpe:>10.2f}")
    print(f"Max Drawdown:      {buyhold_max_dd:>10.2f}%")

    print(f"\n🎯 RELATIVE PERFORMANCE")
    print("-" * 70)
    outperformance = strategy_total_return - buyhold_total_return
    print(f"Outperformance:    {outperformance:>10.2f}%")
    print(
        f"Status:            {'✓ BEATING MARKET' if outperformance > 0 else '✗ UNDERPERFORMING'}"
    )

    # Plot equity curves
    plot_equity_curves(dates_clean, strategy_equity, buyhold_equity, results_dir)

    # Return results
    results = {
        "strategy_return": strategy_total_return,
        "buyhold_return": buyhold_total_return,
        "strategy_sharpe": strategy_sharpe,
        "buyhold_sharpe": buyhold_sharpe,
        "strategy_max_dd": strategy_max_dd,
        "buyhold_max_dd": buyhold_max_dd,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "outperformance": outperformance,
    }

    return results


def plot_equity_curves(dates, strategy_equity, buyhold_equity, results_dir):
    """
    Plot equity curves comparing strategy vs buy-and-hold.

    Args:
        dates (Index): Date index
        strategy_equity (array): Strategy equity curve
        buyhold_equity (array): Buy & hold equity curve
        results_dir (Path): Directory to save plot

    Returns:
        Path: Path to saved plot
    """
    print(f"\n📊 Generating equity curve plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Equity curves
    ax1.plot(
        dates,
        strategy_equity,
        label="ML Strategy (Long-Only)",
        linewidth=2,
        color="#2E86AB",
        alpha=0.9,
    )
    ax1.plot(
        dates,
        buyhold_equity,
        label="Buy & Hold (Benchmark)",
        linewidth=2,
        color="#A23B72",
        alpha=0.9,
        linestyle="--",
    )

    ax1.set_title(
        "Backtesting: Trading Strategy vs Buy & Hold\nStarting Capital: $1,000",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(dates[0], dates[-1])

    # Add performance annotations
    final_strategy = strategy_equity[-1]
    final_buyhold = buyhold_equity[-1]
    ax1.annotate(
        f"${final_strategy:.0f}",
        xy=(dates[-1], final_strategy),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#2E86AB",
    )
    ax1.annotate(
        f"${final_buyhold:.0f}",
        xy=(dates[-1], final_buyhold),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#A23B72",
    )

    # Plot 2: Drawdowns
    strategy_cummax = np.maximum.accumulate(strategy_equity)
    strategy_drawdown = (strategy_equity - strategy_cummax) / strategy_cummax * 100

    buyhold_cummax = np.maximum.accumulate(buyhold_equity)
    buyhold_drawdown = (buyhold_equity - buyhold_cummax) / buyhold_cummax * 100

    ax2.fill_between(
        dates,
        strategy_drawdown,
        0,
        label="ML Strategy Drawdown",
        color="#2E86AB",
        alpha=0.3,
    )
    ax2.fill_between(
        dates,
        buyhold_drawdown,
        0,
        label="Buy & Hold Drawdown",
        color="#A23B72",
        alpha=0.3,
    )

    ax2.set_title("Drawdown Analysis", fontsize=12, fontweight="bold", pad=15)
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(dates[0], dates[-1])

    plt.tight_layout()

    # Save plot
    plot_path = results_dir / "backtest_strategy.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Equity curve saved to: {plot_path}")

    return plot_path


def save_evaluation_report(metrics, backtest_results, results_dir):
    """
    Save comprehensive evaluation report to text file.

    Args:
        metrics (dict): Classification metrics
        backtest_results (dict): Backtesting results
        results_dir (Path): Directory to save report

    Returns:
        Path: Path to saved report
    """
    report_path = results_dir / "evaluation_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GOLD PRICE PREDICTION - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Classification metrics
        f.write("CLASSIFICATION PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")

        # Backtesting results
        f.write("BACKTESTING RESULTS (Test Set)\n")
        f.write("-" * 70 + "\n\n")

        f.write("ML Strategy (Long-Only):\n")
        f.write(f"  Total Return:      {backtest_results['strategy_return']:.2f}%\n")
        f.write(f"  Sharpe Ratio:      {backtest_results['strategy_sharpe']:.2f}\n")
        f.write(f"  Max Drawdown:      {backtest_results['strategy_max_dd']:.2f}%\n")
        f.write(f"  Win Rate:          {backtest_results['win_rate']:.2f}%\n")
        f.write(f"  Total Trades:      {backtest_results['total_trades']}\n\n")

        f.write("Buy & Hold (Benchmark):\n")
        f.write(f"  Total Return:      {backtest_results['buyhold_return']:.2f}%\n")
        f.write(f"  Sharpe Ratio:      {backtest_results['buyhold_sharpe']:.2f}\n")
        f.write(f"  Max Drawdown:      {backtest_results['buyhold_max_dd']:.2f}%\n\n")

        f.write("Relative Performance:\n")
        f.write(f"  Outperformance:    {backtest_results['outperformance']:.2f}%\n")
        f.write(
            f"  Status:            {'BEATING MARKET' if backtest_results['outperformance'] > 0 else 'UNDERPERFORMING'}\n\n"
        )

        f.write("=" * 70 + "\n")
        f.write("Note: Test set = Last 20% of data (temporal split)\n")
        f.write("Strategy: Long-Only (only buy when predicting UP)\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Evaluation report saved to: {report_path}")

    return report_path


def main():
    """
    Main evaluation pipeline.
    """
    print("=" * 70)
    print("GOLD PRICE FORECASTING - MODEL EVALUATION & BACKTESTING")
    print("=" * 70)

    # Setup paths
    model_path, data_path, results_dir = setup_paths()

    # Load model and data
    X, y, gold_returns, dates = load_and_prepare_data(data_path)

    # Create temporal split (last 20% for testing)
    X_train, X_test, y_train, y_test, returns_test, dates_test = create_temporal_split(
        X, y, gold_returns, dates, test_size=0.2
    )

    print("\n🔧 Training evaluation model on TRAIN SET only...")

    # Train a fresh model only on the training set to avoid data leakage (look-ahead bias)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("✓ Evaluation model trained (It has NOT seen the test data)")
    # Evaluate classification performance
    y_pred, metrics = evaluate_classification_metrics(model, X_test, y_test, results_dir)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, results_dir)

    # Backtest trading strategy
    backtest_results = backtest_strategy(
        y_test, y_pred, returns_test, dates_test, results_dir
    )

    # Save comprehensive report
    save_evaluation_report(metrics, backtest_results, results_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n📁 Output Files:")
    print(f"  • Confusion Matrix:    results/confusion_matrix.png")
    print(f"  • Equity Curve:        results/backtest_strategy.png")
    print(f"  • Evaluation Report:   results/evaluation_report.txt")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
