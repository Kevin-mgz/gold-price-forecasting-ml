"""
Gold Price Forecasting - Feature Engineering Script
Processes raw data and creates ML-ready features with proper time-series handling
Author: Murengezi Kevin
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def setup_directories():
    """
    Setup input and output directory paths.
    Creates the processed data directory if it doesn't exist.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"

    # Create processed directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Raw data directory: {raw_data_dir}")
    print(f"✓ Processed data directory: {processed_data_dir}")

    return raw_data_dir, processed_data_dir


def load_and_resample_data(raw_data_dir):
    """
    Load all CSV files and resample to weekly frequency (Friday).

    Args:
        raw_data_dir (Path): Directory containing raw CSV files

    Returns:
        pd.DataFrame: Merged weekly dataframe
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING & RESAMPLING DATA")
    print("=" * 70)

    # ========================================================================
    # Load Gold Price (Target base)
    # ========================================================================
    print("\n⏳ Loading gold_price.csv...")
    gold = pd.read_csv(raw_data_dir / "gold_price.csv")

    # yfinance CSV format: 'Price' column contains dates, first row is 'Ticker'
    # Skip the first row if it contains 'Ticker'
    if gold.iloc[0, 0] == "Ticker":
        gold = gold.iloc[1:].reset_index(drop=True)

    # Set 'Price' column (which contains dates) as index
    if "Price" in gold.columns:
        gold.set_index("Price", inplace=True)
    elif "Date" in gold.columns:
        gold.set_index("Date", inplace=True)
    else:
        gold.set_index(gold.columns[0], inplace=True)

    # Remove rows where index is 'Date' (header row)
    gold = gold[gold.index != "Date"]

    gold.index = pd.to_datetime(gold.index)  # Explicitly convert to DatetimeIndex
    gold = gold[["Close"]].rename(columns={"Close": "Gold_Close"})
    gold["Gold_Close"] = pd.to_numeric(
        gold["Gold_Close"], errors="coerce"
    )  # Ensure numeric
    gold_weekly = gold.resample("W-FRI").last()
    print(f"✓ Gold: {len(gold)} daily → {len(gold_weekly)} weekly records")

    # ========================================================================
    # Load 10-Year Treasury Yield (Nominal Rates)
    # ========================================================================
    print("\n⏳ Loading rates_10y.csv...")
    rates_10y = pd.read_csv(raw_data_dir / "rates_10y.csv")

    # Skip ticker row if present
    if rates_10y.iloc[0, 0] == "Ticker":
        rates_10y = rates_10y.iloc[1:].reset_index(drop=True)

    if "Price" in rates_10y.columns:
        rates_10y.set_index("Price", inplace=True)
    elif "Date" in rates_10y.columns:
        rates_10y.set_index("Date", inplace=True)
    else:
        rates_10y.set_index(rates_10y.columns[0], inplace=True)

    # Remove rows where index is 'Date' (header row)
    rates_10y = rates_10y[rates_10y.index != "Date"]

    rates_10y.index = pd.to_datetime(rates_10y.index)
    rates_10y = rates_10y[["Close"]].rename(columns={"Close": "Rates_10Y"})
    rates_10y["Rates_10Y"] = pd.to_numeric(rates_10y["Rates_10Y"], errors="coerce")
    rates_10y_weekly = rates_10y.resample("W-FRI").last()
    print(
        f"✓ 10Y Rates: {len(rates_10y)} daily → {len(rates_10y_weekly)} weekly records"
    )

    # ========================================================================
    # Load US Dollar Index (DXY)
    # ========================================================================
    print("\n⏳ Loading dxy.csv...")
    dxy = pd.read_csv(raw_data_dir / "dxy.csv")

    # Skip ticker row if present
    if dxy.iloc[0, 0] == "Ticker":
        dxy = dxy.iloc[1:].reset_index(drop=True)

    if "Price" in dxy.columns:
        dxy.set_index("Price", inplace=True)
    elif "Date" in dxy.columns:
        dxy.set_index("Date", inplace=True)
    else:
        dxy.set_index(dxy.columns[0], inplace=True)

    # Remove rows where index is 'Date' (header row)
    dxy = dxy[dxy.index != "Date"]

    dxy.index = pd.to_datetime(dxy.index)
    dxy = dxy[["Close"]].rename(columns={"Close": "DXY_Close"})
    dxy["DXY_Close"] = pd.to_numeric(dxy["DXY_Close"], errors="coerce")
    dxy_weekly = dxy.resample("W-FRI").last()
    print(f"✓ DXY: {len(dxy)} daily → {len(dxy_weekly)} weekly records")

    # ========================================================================
    # Load GLD ETF (Volume as sentiment indicator)
    # ========================================================================
    print("\n⏳ Loading etf_gld.csv...")
    gld = pd.read_csv(raw_data_dir / "etf_gld.csv")

    # Skip ticker row if present
    if gld.iloc[0, 0] == "Ticker":
        gld = gld.iloc[1:].reset_index(drop=True)

    if "Price" in gld.columns:
        gld.set_index("Price", inplace=True)
    elif "Date" in gld.columns:
        gld.set_index("Date", inplace=True)
    else:
        gld.set_index(gld.columns[0], inplace=True)

    # Remove rows where index is 'Date' (header row)
    gld = gld[gld.index != "Date"]

    gld.index = pd.to_datetime(gld.index)
    gld = gld[["Volume"]].rename(columns={"Volume": "GLD_Volume"})
    gld["GLD_Volume"] = pd.to_numeric(gld["GLD_Volume"], errors="coerce")
    gld_weekly = gld.resample("W-FRI").sum()  # Sum volume over the week
    print(f"✓ GLD Volume: {len(gld)} daily → {len(gld_weekly)} weekly records")

    # ========================================================================
    # Load Federal Funds Rate
    # ========================================================================
    print("\n⏳ Loading fed_funds.csv...")
    fed_funds = pd.read_csv(raw_data_dir / "fed_funds.csv")

    if "DATE" in fed_funds.columns:
        fed_funds.set_index("DATE", inplace=True)
    elif "Date" in fed_funds.columns:
        fed_funds.set_index("Date", inplace=True)
    else:
        fed_funds.set_index(fed_funds.columns[0], inplace=True)

    fed_funds.index = pd.to_datetime(fed_funds.index)
    fed_funds = fed_funds.rename(columns={"DFF": "Fed_Funds"})
    fed_funds_weekly = fed_funds.resample("W-FRI").last()
    print(
        f"✓ Fed Funds: {len(fed_funds)} daily → {len(fed_funds_weekly)} weekly records"
    )

    # ========================================================================
    # Load 10-Year Real Interest Rate (TIPS)
    # ========================================================================
    print("\n⏳ Loading real_rates_10y.csv...")
    real_rates = pd.read_csv(raw_data_dir / "real_rates_10y.csv")

    if "DATE" in real_rates.columns:
        real_rates.set_index("DATE", inplace=True)
    elif "Date" in real_rates.columns:
        real_rates.set_index("Date", inplace=True)
    else:
        real_rates.set_index(real_rates.columns[0], inplace=True)

    real_rates.index = pd.to_datetime(real_rates.index)
    real_rates = real_rates.rename(columns={"DFII10": "Real_Rates_10Y"})
    real_rates_weekly = real_rates.resample("W-FRI").last()
    print(
        f"✓ Real Rates: {len(real_rates)} daily → {len(real_rates_weekly)} weekly records"
    )

    # ========================================================================
    # Load Central Bank Demand (Quarterly data - needs forward fill)
    # ========================================================================
    print("\n⏳ Loading central_bank_demand.csv...")
    cb_demand = pd.read_csv(raw_data_dir / "central_bank_demand.csv")

    if "Date" in cb_demand.columns:
        cb_demand.set_index("Date", inplace=True)
    elif "DATE" in cb_demand.columns:
        cb_demand.set_index("DATE", inplace=True)
    else:
        cb_demand.set_index(cb_demand.columns[0], inplace=True)

    cb_demand.index = pd.to_datetime(cb_demand.index)

    # Check column names and rename appropriately
    if "Demand" in cb_demand.columns:
        cb_demand = cb_demand[["Demand"]].rename(columns={"Demand": "CB_Demand"})
    elif "Value" in cb_demand.columns:
        cb_demand = cb_demand[["Value"]].rename(columns={"Value": "CB_Demand"})
    else:
        # Take the first column if column names are different
        cb_demand = cb_demand.iloc[:, [0]].rename(
            columns={cb_demand.columns[0]: "CB_Demand"}
        )

    cb_demand_weekly = cb_demand.resample("W-FRI").last()
    # Forward fill quarterly data to propagate across weeks
    cb_demand_weekly = cb_demand_weekly.fillna(method="ffill")
    print(
        f"✓ CB Demand: {len(cb_demand)} quarterly → {len(cb_demand_weekly)} weekly records (ffill)"
    )

    # ========================================================================
    # Merge all dataframes on date index
    # ========================================================================
    print("\n⏳ Merging all datasets...")
    df = gold_weekly.copy()
    df = df.join(rates_10y_weekly, how="left")
    df = df.join(dxy_weekly, how="left")
    df = df.join(gld_weekly, how="left")
    df = df.join(fed_funds_weekly, how="left")
    df = df.join(real_rates_weekly, how="left")
    df = df.join(cb_demand_weekly, how="left")

    # Forward fill any remaining NaN values from joins
    df = df.fillna(method="ffill")

    print(f"✓ Merged dataset: {len(df)} weekly records")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df


def create_features(df):
    """
    Create engineered features from raw data.

    Args:
        df (pd.DataFrame): Merged weekly dataframe

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    df_features = df.copy()

    # ========================================================================
    # 1. RETURNS (Log Returns for price-based features)
    # ========================================================================
    print("\n⏳ Creating return features...")

    # Gold returns
    df_features["Gold_Return"] = np.log(
        df_features["Gold_Close"] / df_features["Gold_Close"].shift(1)
    )

    # DXY returns
    df_features["DXY_Return"] = np.log(
        df_features["DXY_Close"] / df_features["DXY_Close"].shift(1)
    )

    # GLD Volume percentage change
    df_features["GLD_Volume_Change"] = df_features["GLD_Volume"].pct_change()

    print("✓ Returns created: Gold_Return, DXY_Return, GLD_Volume_Change")

    # ========================================================================
    # 2. RATE CHANGES (First difference for interest rates)
    # ========================================================================
    print("\n⏳ Creating rate change features...")

    # Nominal rate change
    df_features["Rates_10Y_Change"] = df_features["Rates_10Y"].diff()

    # Real rate change
    df_features["Real_Rates_10Y_Change"] = df_features["Real_Rates_10Y"].diff()

    # Fed funds rate change
    df_features["Fed_Funds_Change"] = df_features["Fed_Funds"].diff()

    print(
        "✓ Rate changes created: Rates_10Y_Change, Real_Rates_10Y_Change, Fed_Funds_Change"
    )

    # ========================================================================
    # 3. VOLATILITY (Rolling standard deviation)
    # ========================================================================
    print("\n⏳ Creating volatility features...")

    # 4-week rolling volatility of gold returns
    df_features["Gold_Volatility_4W"] = (
        df_features["Gold_Return"].rolling(window=4).std()
    )

    print("✓ Volatility created: Gold_Volatility_4W")

    # ========================================================================
    # 4. MOMENTUM INDICATORS
    # ========================================================================
    print("\n⏳ Creating momentum features...")

    # Simple Moving Average (15 weeks)
    df_features["Gold_SMA_15"] = df_features["Gold_Close"].rolling(window=15).mean()

    # Price relative to SMA (above or below trend)
    df_features["Gold_Price_to_SMA"] = (
        df_features["Gold_Close"] / df_features["Gold_SMA_15"]
    )

    # Simple RSI proxy (percentage of up weeks in last 14 weeks)
    def calculate_simple_rsi(prices, period=14):
        """Calculate a simplified RSI-like indicator"""
        returns = prices.diff()
        up_weeks = (returns > 0).rolling(window=period).sum()
        rsi_proxy = up_weeks / period * 100
        return rsi_proxy

    df_features["Gold_RSI_Proxy"] = calculate_simple_rsi(
        df_features["Gold_Close"], period=14
    )

    print("✓ Momentum created: Gold_SMA_15, Gold_Price_to_SMA, Gold_RSI_Proxy")

    # ========================================================================
    # 5. CENTRAL BANK DEMAND (Already weekly, keep as-is)
    # ========================================================================
    print("\n⏳ Keeping CB_Demand as feature...")
    print("✓ CB_Demand retained (forward-filled quarterly data)")

    return df_features


def create_target(df):
    """
    Create binary target variable: direction of gold price next week.
    Target = 1 if next week's close > this week's close, else 0

    Args:
        df (pd.DataFrame): Dataframe with features

    Returns:
        pd.DataFrame: Dataframe with target variable added
    """
    print("\n" + "=" * 70)
    print("STEP 3: TARGET CREATION")
    print("=" * 70)

    print("\n⏳ Creating binary target (next week direction)...")

    # Calculate next week's closing price
    df["Gold_Close_Next"] = df["Gold_Close"].shift(-1)

    # Create binary target: 1 if price goes up, 0 if down
    df["Target"] = (df["Gold_Close_Next"] > df["Gold_Close"]).astype(int)

    # Drop the helper column
    df = df.drop(columns=["Gold_Close_Next"])

    print(f"✓ Target created: Target (binary)")
    print(f"  Class distribution:")
    print(
        f"    Up weeks (1): {df['Target'].sum()} ({df['Target'].sum()/len(df)*100:.1f}%)"
    )
    print(
        f"    Down weeks (0): {(df['Target']==0).sum()} ({(df['Target']==0).sum()/len(df)*100:.1f}%)"
    )

    return df


def prevent_data_leakage(df):
    """
    Shift all features by 1 week to prevent data leakage.
    At time t, we should only have information available up to time t
    to predict the target at time t (which represents t->t+1 direction).

    Args:
        df (pd.DataFrame): Dataframe with features and target

    Returns:
        pd.DataFrame: Dataframe with shifted features
    """
    print("\n" + "=" * 70)
    print("STEP 4: PREVENTING DATA LEAKAGE")
    print("=" * 70)

    print("\n⏳ Shifting features by 1 week...")

    # List of feature columns to shift (everything except raw prices and target)
    feature_cols = [
        "Gold_Return",
        "DXY_Return",
        "GLD_Volume_Change",
        "Rates_10Y_Change",
        "Real_Rates_10Y_Change",
        "Fed_Funds_Change",
        "Gold_Volatility_4W",
        "Gold_SMA_15",
        "Gold_Price_to_SMA",
        "Gold_RSI_Proxy",
        "CB_Demand",
    ]

    # Also shift the base level variables used for features
    base_cols = ["Rates_10Y", "Real_Rates_10Y", "Fed_Funds", "DXY_Close", "GLD_Volume"]

    all_shift_cols = feature_cols + base_cols

    # Shift all feature columns by 1 week
    for col in all_shift_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)

    print(f"✓ Shifted {len(all_shift_cols)} feature columns by 1 week")
    print(
        "  Now at time t, we have data from t-1 to predict target at t (t->t+1 direction)"
    )

    return df


def clean_and_finalize(df):
    """
    Remove rows with NaN values and select final feature set.

    Args:
        df (pd.DataFrame): Dataframe with all features

    Returns:
        pd.DataFrame: Clean final dataset
    """
    print("\n" + "=" * 70)
    print("STEP 5: CLEANING & FINALIZATION")
    print("=" * 70)

    print(f"\n⏳ Initial dataset size: {len(df)} rows")
    print(f"   NaN values: {df.isna().sum().sum()}")

    # Drop rows with any NaN values
    df_clean = df.dropna()

    print(f"✓ After dropping NaN: {len(df_clean)} rows")
    print(f"   Rows removed: {len(df) - len(df_clean)}")

    # Select final feature set for modeling
    feature_cols = [
        # Returns
        "Gold_Return",
        "DXY_Return",
        "GLD_Volume_Change",
        # Rate changes
        "Rates_10Y_Change",
        "Real_Rates_10Y_Change",
        "Fed_Funds_Change",
        # Volatility
        "Gold_Volatility_4W",
        # Momentum
        "Gold_Price_to_SMA",
        "Gold_RSI_Proxy",
        # Fundamental
        "CB_Demand",
        # Target
        "Target",
    ]

    # Also keep some raw values for reference
    reference_cols = ["Gold_Close", "DXY_Close", "Rates_10Y", "Real_Rates_10Y"]

    final_cols = reference_cols + feature_cols
    df_final = df_clean[final_cols].copy()

    print(f"\n✓ Final dataset ready!")
    print(f"   Shape: {df_final.shape}")
    print(f"   Features: {len(feature_cols) - 1}")  # -1 for target
    print(f"   Date range: {df_final.index[0].date()} to {df_final.index[-1].date()}")

    return df_final


def display_summary(df, processed_data_dir):
    """
    Display summary statistics and correlation with target.

    Args:
        df (pd.DataFrame): Final clean dataset
        processed_data_dir (Path): Directory to save the file
    """
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print("\n📊 First 5 rows:")
    print(df.head())

    print("\n📊 Last 5 rows:")
    print(df.tail())

    print("\n📊 Dataset Info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total features: {df.shape[1] - 1}")  # -1 for target
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Correlation with target
    print("\n📊 Feature Correlation with Target (sorted):")
    feature_cols = [
        col
        for col in df.columns
        if col
        not in ["Target", "Gold_Close", "DXY_Close", "Rates_10Y", "Real_Rates_10Y"]
    ]
    correlations = (
        df[feature_cols + ["Target"]]
        .corr()["Target"]
        .drop("Target")
        .sort_values(ascending=False)
    )

    print("\nTop Positive Correlations:")
    print(correlations.head(5))

    print("\nTop Negative Correlations:")
    print(correlations.tail(5))

    # Save to CSV
    output_path = processed_data_dir / "dataset_final.csv"
    df.to_csv(output_path)
    print(f"\n✅ Dataset saved to: {output_path}")


def main():
    """
    Main function to orchestrate the feature engineering pipeline.
    """
    print("=" * 70)
    print("GOLD PRICE FORECASTING - FEATURE ENGINEERING")
    print("=" * 70)

    # Setup directories
    raw_data_dir, processed_data_dir = setup_directories()

    # Step 1: Load and resample data
    df = load_and_resample_data(raw_data_dir)

    # Step 2: Create features
    df = create_features(df)

    # Step 3: Create target
    df = create_target(df)

    # Step 4: Prevent data leakage
    df = prevent_data_leakage(df)

    # Step 5: Clean and finalize
    df_final = clean_and_finalize(df)

    # Display summary and save
    display_summary(df_final, processed_data_dir)

    print("\n" + "=" * 70)
    print("🎉 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the correlation matrix")
    print("  2. Consider additional feature selection if needed")
    print("  3. Proceed to model training")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
