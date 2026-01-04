"""
Gold Price Forecasting - Modeling Script
Trains and evaluates Random Forest classifier with time series cross-validation
Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def setup_directories():
    """
    Create necessary directories for models and results if they don't exist.

    Returns:
        tuple: (models_dir, results_dir) Path objects
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    models_dir = project_root / "models"
    results_dir = project_root / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Models directory: {models_dir}")
    print(f"✓ Results directory: {results_dir}")

    return models_dir, results_dir


def load_data():
    """
    Load the processed dataset and separate features from target.

    Returns:
        tuple: (X, y, feature_names, data_path)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "processed" / "dataset_final.csv"

    print(f"\n📂 Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset with date as index
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(
        f"  Index: {df.index.name} (from {df.index[0].date()} to {df.index[-1].date()})"
    )

    # Define reference columns to exclude from features
    reference_columns = ["Gold_Close", "DXY_Close", "Rates_10Y", "Real_Rates_10Y"]
    target_column = "Target"

    # Check if target exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Get all columns except target and reference columns
    exclude_columns = reference_columns + [target_column]
    potential_features = [col for col in df.columns if col not in exclude_columns]

    # Filter only numeric columns
    numeric_features = []
    non_numeric_cols = []

    for col in potential_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            non_numeric_cols.append(col)

    # Alert if non-numeric columns found
    if non_numeric_cols:
        print(f"\n⚠️  Warning: Excluding {len(non_numeric_cols)} non-numeric column(s):")
        for col in non_numeric_cols:
            print(f"    - {col} (dtype: {df[col].dtype})")

    # Create feature matrix and target vector
    X = df[numeric_features].copy()
    y = df[target_column].copy()

    print(f"\n✓ Features (X): {X.shape[1]} columns")
    print(f"  Feature list: {numeric_features}")
    print(f"✓ Target (y): {y.shape[0]} samples")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    return X, y, numeric_features, data_path


def time_series_cross_validation(X, y, n_splits=5):
    """
    Perform time series cross-validation with Random Forest.

    Args:
        X (DataFrame): Feature matrix
        y (Series): Target vector
        n_splits (int): Number of splits for TimeSeriesSplit

    Returns:
        dict: Dictionary containing metrics for each fold
    """
    print("\n" + "=" * 70)
    print("TIME SERIES CROSS-VALIDATION")
    print("=" * 70)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Storage for metrics
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    # Iterate through splits
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\n📊 Fold {fold}/{n_splits}")
        print("-" * 70)

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set:  {len(X_test)} samples")

        # Initialize and train model
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Store metrics
        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)

        # Display fold results
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    return metrics


def display_metrics_summary(metrics):
    """
    Display summary statistics of cross-validation metrics.

    Args:
        metrics (dict): Dictionary containing metric lists

    Returns:
        dict: Dictionary with mean and std for each metric
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    summary = {}

    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric_name] = {"mean": mean_val, "std": std_val}

        print(f"\n{metric_name.upper()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")

    return summary


def train_final_model(X, y):
    """
    Train final model on the entire dataset.

    Args:
        X (DataFrame): Complete feature matrix
        y (Series): Complete target vector

    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    print(f"\n🔧 Training on full dataset ({len(X)} samples)...")

    # Initialize model with same parameters
    final_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    # Train on all data
    final_model.fit(X, y)

    print("✓ Final model training complete")

    return final_model


def save_model(model, models_dir):
    """
    Save trained model to disk.

    Args:
        model: Trained sklearn model
        models_dir (Path): Directory to save the model

    Returns:
        Path: Path to saved model
    """
    model_path = models_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)

    print(f"✓ Model saved to: {model_path}")

    return model_path


def plot_feature_importance(model, feature_names, results_dir):
    """
    Create and save feature importance visualization.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        results_dir (Path): Directory to save the plot

    Returns:
        Path: Path to saved plot
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame for easier manipulation
    feature_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=True)

    # Display top 10 features
    print("\n🔝 Top 10 Most Important Features:")
    print("-" * 70)
    top_features = feature_imp_df.tail(10).sort_values("Importance", ascending=False)
    for idx, row in top_features.iterrows():
        print(f"  {row['Feature']:30s} : {row['Importance']:.4f}")

    # Create horizontal bar plot
    plt.figure(figsize=(12, 10))

    # Plot all features (or top 20 if too many)
    n_features = min(20, len(feature_imp_df))
    plot_data = feature_imp_df.tail(n_features)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_data)))

    plt.barh(range(len(plot_data)), plot_data["Importance"], color=colors)
    plt.yticks(range(len(plot_data)), plot_data["Feature"])
    plt.xlabel("Feature Importance", fontsize=12, fontweight="bold")
    plt.ylabel("Features", fontsize=12, fontweight="bold")
    plt.title(
        "Random Forest Feature Importance\nTop 20 Features",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = results_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Feature importance plot saved to: {plot_path}")

    return plot_path


def save_metrics_report(metrics_summary, results_dir):
    """
    Save metrics summary to text file.

    Args:
        metrics_summary (dict): Dictionary with metric statistics
        results_dir (Path): Directory to save the report

    Returns:
        Path: Path to saved report
    """
    report_path = results_dir / "metrics.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GOLD PRICE PREDICTION - MODEL PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL: Random Forest Classifier\n")
        f.write("  - n_estimators: 100\n")
        f.write("  - max_depth: 10\n")
        f.write("  - random_state: 42\n\n")

        f.write("CROSS-VALIDATION: Time Series Split (5 folds)\n\n")

        f.write("-" * 70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n\n")

        for metric_name, stats in metrics_summary.items():
            f.write(f"{metric_name.upper()}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std:  {stats['std']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Note: Model trained on time series data with temporal validation.\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Metrics report saved to: {report_path}")

    return report_path


def main():
    """
    Main function to orchestrate the modeling pipeline.
    """
    print("=" * 70)
    print("GOLD PRICE FORECASTING - MODELING PHASE")
    print("=" * 70)

    # Setup
    models_dir, results_dir = setup_directories()

    # Load data
    X, y, feature_names, data_path = load_data()

    # Time series cross-validation
    metrics = time_series_cross_validation(X, y, n_splits=5)

    # Display summary
    metrics_summary = display_metrics_summary(metrics)

    # Train final model
    final_model = train_final_model(X, y)

    # Save model
    model_path = save_model(final_model, models_dir)

    # Create visualizations
    plot_path = plot_feature_importance(final_model, feature_names, results_dir)

    # Save metrics report
    report_path = save_metrics_report(metrics_summary, results_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ MODELING PHASE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n📊 Model Performance:")
    print(
        f"  Accuracy:  {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}"
    )
    print(
        f"  Precision: {metrics_summary['precision']['mean']:.4f} ± {metrics_summary['precision']['std']:.4f}"
    )
    print(
        f"  Recall:    {metrics_summary['recall']['mean']:.4f} ± {metrics_summary['recall']['std']:.4f}"
    )
    print(
        f"  F1-Score:  {metrics_summary['f1']['mean']:.4f} ± {metrics_summary['f1']['std']:.4f}"
    )

    print(f"\n📁 Output Files:")
    print(f"  Model:      {model_path}")
    print(f"  Plot:       {plot_path}")
    print(f"  Report:     {report_path}")

    print("\n🎯 Next Step: Run evaluation on test set or deploy the model!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
