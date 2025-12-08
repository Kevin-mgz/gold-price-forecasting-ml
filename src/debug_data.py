"""
Gold Price Forecasting - Data Leakage Diagnostic Script
Investigates suspicious correlations and potential look-ahead bias
Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_alert(message):
    """Print alert message (simulated red text with symbols)."""
    print(f"\n🚨 ⚠️  ALERT: {message} ⚠️  🚨\n")


def load_dataset():
    """
    Load the processed dataset.

    Returns:
        DataFrame: Loaded dataset
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "processed" / "dataset_final.csv"

    print(f"📂 Loading dataset from: {data_path}\n")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load with date as index
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df


def display_columns(df):
    """
    Display all columns in the dataset.

    Args:
        df (DataFrame): Dataset
    """
    print_header("DATASET COLUMNS")

    print(f"Total columns: {len(df.columns)}\n")

    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        print(f"{i:2d}. {col:30s} | dtype: {str(dtype):10s} | nulls: {null_count}")


def analyze_correlations(df):
    """
    Calculate and display correlation matrix with Target.

    Args:
        df (DataFrame): Dataset

    Returns:
        Series: Correlations with Target
    """
    print_header("CORRELATION ANALYSIS WITH TARGET")

    # Check if Target exists
    if "Target" not in df.columns:
        print("❌ ERROR: 'Target' column not found in dataset")
        return None

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Analyzing {len(numeric_cols)} numeric columns...\n")

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Get correlations with Target
    target_corr = corr_matrix["Target"].drop("Target")  # Remove self-correlation

    # Sort by absolute value (strongest correlations first)
    target_corr_sorted = target_corr.reindex(
        target_corr.abs().sort_values(ascending=False).index
    )

    return target_corr_sorted


def display_correlations(target_corr, threshold=0.8):
    """
    Display correlations and highlight suspicious ones.

    Args:
        target_corr (Series): Correlations with Target
        threshold (float): Threshold for suspicious correlation
    """
    print("-" * 80)
    print(f"{'Feature':<35} | {'Correlation':>12} | {'Status'}")
    print("-" * 80)

    suspicious_features = []

    for feature, corr_value in target_corr.items():
        abs_corr = abs(corr_value)

        # Determine status
        if abs_corr >= threshold:
            status = "🚨 SUSPICIOUS!"
            suspicious_features.append((feature, corr_value))
        elif abs_corr >= 0.5:
            status = "⚠️  High"
        elif abs_corr >= 0.3:
            status = "📊 Moderate"
        else:
            status = "✓ Low"

        print(f"{feature:<35} | {corr_value:>12.4f} | {status}")

    return suspicious_features


def display_suspicious_summary(suspicious_features, threshold=0.8):
    """
    Display summary of suspicious features with detailed analysis.

    Args:
        suspicious_features (list): List of (feature, correlation) tuples
        threshold (float): Correlation threshold
    """
    print_header("🚨 DATA LEAKAGE DETECTION SUMMARY")

    if not suspicious_features:
        print("✅ No suspicious correlations detected!")
        print(f"   All features have |correlation| < {threshold}")
        return

    print_alert(f"FOUND {len(suspicious_features)} SUSPICIOUS FEATURE(S)")

    print("These features have abnormally high correlation with Target:")
    print("This suggests potential DATA LEAKAGE or LOOK-AHEAD BIAS!\n")

    print("-" * 80)
    print(f"{'Feature':<35} | {'Correlation':>12} | {'Risk Level'}")
    print("-" * 80)

    for feature, corr_value in suspicious_features:
        abs_corr = abs(corr_value)

        if abs_corr >= 0.95:
            risk = "🔴 CRITICAL - Likely contains future info"
        elif abs_corr >= 0.85:
            risk = "🟠 HIGH - Strong leak suspected"
        else:
            risk = "🟡 MEDIUM - Investigate further"

        print(f"{feature:<35} | {corr_value:>12.4f} | {risk}")

    print("-" * 80)

    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 80)
    for i, (feature, corr_value) in enumerate(suspicious_features, 1):
        print(f"\n{i}. {feature} (corr = {corr_value:.4f})")
        print(f"   ⚠️  ACTION: Remove this feature from your model")
        print(f"   ❓ WHY: Correlation > {threshold} indicates it contains")
        print(f"          information about the target that wouldn't be")
        print(f"          available at prediction time (look-ahead bias)")

        # Specific recommendations based on feature name
        if "return" in feature.lower() or "pct" in feature.lower():
            print(f"   💡 HINT: Return-based features should be lagged or use")
            print(f"          past values only, never current period returns")
        elif "future" in feature.lower() or "next" in feature.lower():
            print(f"   💡 HINT: This appears to be a future-looking variable")
        elif "target" in feature.lower():
            print(f"   💡 HINT: This might be derived directly from the target")


def display_model_performance_explanation():
    """Display explanation of the accuracy/performance paradox."""
    print_header("WHY HIGH ACCURACY BUT POOR BACKTESTING?")

    print("🎯 THE PARADOX EXPLAINED:")
    print("-" * 80)
    print("Your model achieves 97% accuracy because it's 'cheating' - it has")
    print("access to information that wouldn't be available in real trading.")
    print()
    print("Example of Data Leakage:")
    print("  ❌ BAD:  Using today's gold return to predict today's direction")
    print("  ❌ BAD:  Using current period's features without lag")
    print("  ✅ GOOD: Using yesterday's data to predict tomorrow's direction")
    print()
    print("When you backtest, the model can't access this leaked information,")
    print("so it performs poorly despite high training accuracy.")
    print()
    print("This is why it's critical to ensure temporal consistency in")
    print("feature engineering for time series problems!")
    print("-" * 80)


def analyze_feature_timing(df):
    """
    Analyze if features are properly lagged.

    Args:
        df (DataFrame): Dataset
    """
    print_header("FEATURE TIMING ANALYSIS")

    print("Checking for potential timing issues...\n")

    # Look for features that might not be lagged
    potentially_problematic = []

    for col in df.columns:
        col_lower = col.lower()

        # Check for problematic patterns
        if any(keyword in col_lower for keyword in ["return", "pct_change", "diff"]):
            if not any(
                lag_indicator in col_lower
                for lag_indicator in ["lag", "prev", "shift", "ma", "ema"]
            ):
                potentially_problematic.append(
                    (col, "Return/change without explicit lag")
                )

        if any(keyword in col_lower for keyword in ["future", "next", "forward"]):
            potentially_problematic.append((col, "Future-looking variable name"))

    if potentially_problematic:
        print("⚠️  Found features that may need review:\n")
        for feature, reason in potentially_problematic:
            print(f"  • {feature:30s} - {reason}")
    else:
        print("✅ No obvious timing issues detected in feature names")


def main():
    """
    Main diagnostic pipeline.
    """
    print("=" * 80)
    print("  🔍 DATA LEAKAGE DIAGNOSTIC TOOL")
    print("  Gold Price Forecasting Project")
    print("=" * 80)

    try:
        # Load dataset
        df = load_dataset()

        # Display all columns
        display_columns(df)

        # Analyze correlations
        target_corr = analyze_correlations(df)

        if target_corr is not None:
            # Display correlations with highlighting
            print()
            suspicious_features = display_correlations(target_corr, threshold=0.8)

            # Display detailed summary of suspicious features
            display_suspicious_summary(suspicious_features, threshold=0.8)

            # Analyze feature timing
            analyze_feature_timing(df)

            # Explain the paradox
            if suspicious_features:
                display_model_performance_explanation()

        # Final summary
        print_header("DIAGNOSTIC COMPLETE")
        print("Next steps:")
        print("1. Remove the suspicious features identified above")
        print("2. Re-run feature engineering with proper lagging")
        print("3. Retrain the model")
        print("4. Re-evaluate with backtesting")
        print()
        print("Expected outcome after fixing:")
        print("  • Lower accuracy (60-70% is realistic for financial markets)")
        print("  • Better backtesting performance (should beat buy-and-hold)")
        print("  • More realistic and deployable model")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()
