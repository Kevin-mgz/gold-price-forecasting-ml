"""
Gold Price Forecasting - Regression Bonus Script
Predicts actual returns (volatility) instead of direction
Author: AI Kevin Murengezi
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def setup_paths():
    """
    Setup project paths.

    Returns:
        tuple: (data_path, results_dir)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_path = project_root / "data" / "processed" / "dataset_final.csv"
    results_dir = project_root / "results"

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    return data_path, results_dir


def load_and_prepare_data(data_path):
    """
    Load dataset and create continuous target variable.

    Args:
        data_path (Path): Path to processed dataset

    Returns:
        tuple: (X, y, dates)
    """
    print_header("DATA LOADING & PREPARATION")

    print(f"📂 Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load with date as index
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Create continuous target: next week's return (%)
    print("\n🎯 Creating regression target...")
    print("   Target = Next week's return (Gold_Close % change)")

    df["Target_Return"] = df["Gold_Close"].pct_change().shift(-1) * 100

    # Remove NaN rows created by shift and pct_change
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)

    print(f"✓ Target variable created: 'Target_Return'")
    print(f"✓ Removed {removed_rows} NaN rows")
    print(f"✓ Final dataset: {df.shape[0]} rows")

    # Display target statistics
    print(f"\n📊 Target Statistics:")
    print(f"   Mean Return:    {df['Target_Return'].mean():>8.3f}%")
    print(f"   Std Dev:        {df['Target_Return'].std():>8.3f}%")
    print(f"   Min Return:     {df['Target_Return'].min():>8.3f}%")
    print(f"   Max Return:     {df['Target_Return'].max():>8.3f}%")

    # Define columns to exclude from features
    exclude_columns = [
        "Target",
        "Target_Return",
        "Gold_Close",
        "DXY_Close",
        "Rates_10Y",
        "Real_Rates_10Y",
    ]

    # Get numeric feature columns
    feature_columns = [
        col
        for col in df.columns
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Prepare features and target
    X = df[feature_columns].copy()
    y = df["Target_Return"].copy()
    dates = df.index

    print(f"\n✓ Features (X): {len(feature_columns)} columns")
    print(f"   {feature_columns}")
    print(f"✓ Target (y): {len(y)} samples")

    return X, y, dates


def temporal_split(X, y, dates, test_size=0.2):
    """
    Split data temporally (chronological split).

    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
        dates (Index): Date index
        test_size (float): Proportion for test set

    Returns:
        tuple: (X_train, X_test, y_train, y_test, dates_train, dates_test)
    """
    print_header("TEMPORAL SPLIT")

    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))

    # Split chronologically
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]

    print(f"📊 Data Split (Temporal):")
    print(
        f"   Train: {len(X_train)} samples ({dates_train[0].date()} to {dates_train[-1].date()})"
    )
    print(
        f"   Test:  {len(X_test)} samples ({dates_test[0].date()} to {dates_test[-1].date()})"
    )
    print(f"   Split: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def train_regression_model(X_train, y_train):
    """
    Train Random Forest Regressor.

    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target

    Returns:
        model: Trained regressor
    """
    print_header("MODEL TRAINING")

    print("🤖 Initializing Random Forest Regressor...")
    print("   Parameters:")
    print("   - n_estimators: 100")
    print("   - random_state: 42")
    print("   - n_jobs: -1 (all cores)")

    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    print(f"\n🔧 Training on {len(X_train)} samples...")

    # Train model
    model.fit(X_train, y_train)

    print("✓ Model training complete!")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate regression model performance.

    Args:
        model: Trained regressor
        X_test (DataFrame): Test features
        y_test (Series): Test target

    Returns:
        tuple: (y_pred, rmse, r2)
    """
    print_header("MODEL EVALUATION")

    print("📊 Making predictions on test set...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    # Display metrics
    print(f"\n📈 REGRESSION METRICS:")
    print("-" * 80)
    print(f"   RMSE (Root Mean Squared Error):  {rmse:.4f}%")
    print(f"   MAE (Mean Absolute Error):        {mae:.4f}%")
    print(f"   R² Score:                         {r2:.4f}")
    print("-" * 80)

    # Interpretation based on R²
    print(f"\n💡 Model Performance Interpretation:")
    if r2 >= 0.50:
        interpretation = "🟢 EXCELLENT - Model captures significant volatility patterns"
        quality = "Strong predictive power"
    elif r2 >= 0.30:
        interpretation = "🟡 GOOD - Model captures moderate volatility patterns"
        quality = "Decent predictive capability"
    elif r2 >= 0.10:
        interpretation = "🟠 FAIR - Model captures some patterns but limited accuracy"
        quality = "Weak but non-random predictions"
    elif r2 >= 0:
        interpretation = "🔴 POOR - Model struggles to predict returns accurately"
        quality = "Very limited predictive power"
    else:
        interpretation = "❌ VERY POOR - Model performs worse than baseline"
        quality = "No predictive value"

    print(f"   {interpretation}")
    print(f"   Quality: {quality}")
    print(
        f"   Explanation: R² = {r2:.4f} means the model explains {r2*100:.1f}% of variance"
    )

    return y_pred, rmse, r2


def plot_predictions(dates_test, y_test, y_pred, rmse, r2, results_dir):
    """
    Create visualization of actual vs predicted returns.

    Args:
        dates_test (Index): Test dates
        y_test (Series): Actual returns
        y_pred (array): Predicted returns
        rmse (float): RMSE score
        r2 (float): R² score
        results_dir (Path): Directory to save plot

    Returns:
        Path: Path to saved plot
    """
    print_header("VISUALIZATION")

    print("📊 Generating forecast plot...")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Time series comparison
    ax1.plot(
        dates_test,
        y_test,
        label="Actual Returns",
        color="#2E86AB",
        linewidth=2,
        alpha=0.8,
    )
    ax1.plot(
        dates_test,
        y_pred,
        label="Predicted Returns",
        color="#FF6B35",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
    )

    ax1.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.set_title(
        "Gold Price Returns: Actual vs Predicted\nRandom Forest Regression",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Weekly Return (%)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = f"RMSE: {rmse:.3f}%\nR²: {r2:.3f}"
    ax1.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Plot 2: Scatter plot (Actual vs Predicted)
    ax2.scatter(y_test, y_pred, alpha=0.5, color="#764BA2", s=50)

    # Add diagonal line (perfect prediction)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax2.set_title(
        "Actual vs Predicted Returns (Scatter)", fontsize=12, fontweight="bold", pad=15
    )
    ax2.set_xlabel("Actual Returns (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Returns (%)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = results_dir / "regression_forecast.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Plot saved to: {plot_path}")

    return plot_path


def display_feature_importance(model, feature_names, top_n=10):
    """
    Display top feature importances for regression.

    Args:
        model: Trained model
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
    """
    print_header("FEATURE IMPORTANCE ANALYSIS")

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame
    feature_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    print(f"🔝 Top {top_n} Most Important Features for Predicting Returns:\n")
    print("-" * 80)

    for i, row in feature_imp_df.head(top_n).iterrows():
        print(f"   {row['Feature']:30s} : {row['Importance']:.4f}")

    print("-" * 80)


def save_results_summary(rmse, r2, results_dir):
    """
    Save regression results to text file.

    Args:
        rmse (float): RMSE score
        r2 (float): R² score
        results_dir (Path): Directory to save report

    Returns:
        Path: Path to saved report
    """
    report_path = results_dir / "regression_results.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GOLD PRICE REGRESSION - VOLATILITY PREDICTION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL: Random Forest Regressor\n")
        f.write("TARGET: Next week's return (% change)\n")
        f.write("SPLIT: 80% Train / 20% Test (Temporal)\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"RMSE (Root Mean Squared Error):  {rmse:.4f}%\n")
        f.write(f"R² Score:                         {r2:.4f}\n\n")

        f.write("INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        if r2 >= 0.30:
            f.write(
                "The model shows good capability in predicting return volatility.\n"
            )
            f.write(f"It explains {r2*100:.1f}% of the variance in gold returns.\n")
        elif r2 >= 0.10:
            f.write("The model captures some patterns but with limited accuracy.\n")
            f.write(
                "Predicting exact returns is inherently difficult in financial markets.\n"
            )
        else:
            f.write("The model struggles to predict exact returns accurately.\n")
            f.write("This is expected as financial returns are highly stochastic.\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("NOTE: Predicting exact returns is significantly harder than\n")
        f.write("      predicting direction (classification). Even low R² values\n")
        f.write("      can be valuable in financial applications.\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results summary saved to: {report_path}")

    return report_path


def main():
    """
    Main regression pipeline.
    """
    print("=" * 80)
    print("  🎯 GOLD PRICE REGRESSION - VOLATILITY PREDICTION")
    print("  Bonus: Predicting Actual Returns (not just direction)")
    print("=" * 80)

    try:
        # Setup paths
        data_path, results_dir = setup_paths()

        # Load and prepare data
        X, y, dates = load_and_prepare_data(data_path)

        # Temporal split
        X_train, X_test, y_train, y_test, dates_train, dates_test = temporal_split(
            X, y, dates, test_size=0.2
        )

        # Train model
        model = train_regression_model(X_train, y_train)

        # Evaluate model
        y_pred, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Display feature importance
        display_feature_importance(model, X.columns.tolist(), top_n=10)

        # Create visualization
        plot_path = plot_predictions(dates_test, y_test, y_pred, rmse, r2, results_dir)

        # Save results summary
        report_path = save_results_summary(rmse, r2, results_dir)

        # Final summary
        print("\n" + "=" * 80)
        print("✅ REGRESSION ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"\n📊 Key Findings:")
        print(f"   RMSE:     {rmse:.4f}%")
        print(f"   R² Score: {r2:.4f}")
        print(f"\n📁 Output Files:")
        print(f"   Plot:   {plot_path}")
        print(f"   Report: {report_path}")

        print(f"\n💡 Insight:")
        print(f"   Predicting exact returns is much harder than predicting direction.")
        print(f"   Financial markets are inherently noisy and stochastic.")
        if r2 > 0.20:
            print(
                f"   Your R² of {r2:.3f} suggests the model captures meaningful patterns!"
            )
        else:
            print(
                f"   Consider this a learning exercise - even hedge funds struggle here!"
            )

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()
