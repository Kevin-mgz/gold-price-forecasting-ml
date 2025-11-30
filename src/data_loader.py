"""
Gold Price Forecasting - Data Loader Script
Downloads historical financial data from Yahoo Finance and FRED
Author: Murengezi Kevin
Date: 2025
"""

import yfinance as yf
import pandas_datareader as pdr
from pathlib import Path
from datetime import datetime
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore")


def setup_directories():
    """
    Create the data/raw directory structure if it doesn't exist.
    Uses pathlib for cross-platform compatibility.
    """
    # Get the script's directory and navigate to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Define the raw data directory path
    raw_data_dir = project_root / "data" / "raw"

    # Create directory if it doesn't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Data directory ready: {raw_data_dir}")
    return raw_data_dir


def download_yahoo_data(ticker, output_filename, data_dir, start_date, end_date):
    """
    Download historical data from Yahoo Finance and save to CSV.

    Args:
        ticker (str): Yahoo Finance ticker symbol
        output_filename (str): Output CSV filename
        data_dir (Path): Directory where to save the file
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n⏳ Downloading {ticker} from Yahoo Finance...")

        # Download data using yfinance
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Check if data is empty
        if data.empty:
            print(f"✗ ERROR: No data retrieved for {ticker}")
            return False

        # Save to CSV
        output_path = data_dir / output_filename
        data.to_csv(output_path)

        print(f"✓ SUCCESS: {output_filename} saved ({len(data)} records)")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")

        return True

    except Exception as e:
        print(f"✗ ERROR downloading {ticker}: {str(e)}")
        return False


def download_fred_data(series_id, output_filename, data_dir, start_date, end_date):
    """
    Download economic data from FRED (Federal Reserve Economic Data) and save to CSV.

    Args:
        series_id (str): FRED series identifier
        output_filename (str): Output CSV filename
        data_dir (Path): Directory where to save the file
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n⏳ Downloading {series_id} from FRED...")

        # Download data using pandas_datareader
        data = pdr.DataReader(series_id, "fred", start_date, end_date)

        # Check if data is empty
        if data.empty:
            print(f"✗ ERROR: No data retrieved for {series_id}")
            return False

        # Save to CSV
        output_path = data_dir / output_filename
        data.to_csv(output_path)

        print(f"✓ SUCCESS: {output_filename} saved ({len(data)} records)")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")

        return True

    except Exception as e:
        print(f"✗ ERROR downloading {series_id}: {str(e)}")
        return False


def main():
    """
    Main function to orchestrate the data download process.
    Downloads all required financial datasets for gold price forecasting.
    """
    print("=" * 70)
    print("GOLD PRICE FORECASTING - DATA LOADER")
    print("=" * 70)

    # Setup directories
    data_dir = setup_directories()

    # Define date range
    start_date = "2005-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"\n📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Frequency: Daily")
    print("\n" + "=" * 70)

    # Track download statistics
    total_downloads = 0
    successful_downloads = 0

    # ========================================================================
    # YAHOO FINANCE DOWNLOADS
    # ========================================================================

    yahoo_datasets = [
        {"ticker": "GC=F", "filename": "gold_price.csv", "description": "Gold Futures"},
        {
            "ticker": "^TNX",
            "filename": "rates_10y.csv",
            "description": "10-Year Treasury Yield",
        },
        {"ticker": "DX-Y.NYB", "filename": "dxy.csv", "description": "US Dollar Index"},
        {
            "ticker": "GLD",
            "filename": "etf_gld.csv",
            "description": "SPDR Gold ETF (Volume focus)",
        },
    ]

    print("\n📈 YAHOO FINANCE DOWNLOADS")
    print("-" * 70)

    for dataset in yahoo_datasets:
        total_downloads += 1
        if download_yahoo_data(
            ticker=dataset["ticker"],
            output_filename=dataset["filename"],
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
        ):
            successful_downloads += 1

    # ========================================================================
    # FRED DOWNLOADS
    # ========================================================================

    fred_datasets = [
        {
            "series_id": "DFF",
            "filename": "fed_funds.csv",
            "description": "Effective Federal Funds Rate (Daily)",
        }
    ]

    print("\n\n🏛️  FRED (FEDERAL RESERVE) DOWNLOADS")
    print("-" * 70)

    for dataset in fred_datasets:
        total_downloads += 1
        if download_fred_data(
            series_id=dataset["series_id"],
            output_filename=dataset["filename"],
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
        ):
            successful_downloads += 1

    # ========================================================================
    # BUNDLED DATA VERIFICATION
    # ========================================================================

    print("\n\n📦 BUNDLED DATA VERIFICATION")
    print("-" * 70)

    # Check for central bank demand data
    bundled_file = "central_bank_demand.csv"
    bundled_path = data_dir / bundled_file

    total_downloads += 1

    if bundled_path.exists():
        print(f"\n✓ SUCCESS: '{bundled_file}' found (bundled)")
        print(f"  Location: {bundled_path}")
        successful_downloads += 1
    else:
        print(f"\n✗ ERROR: File missing - '{bundled_file}'")
        print(f"  Expected location: {bundled_path}")
        print(f"  This file should be included in the repository.")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✓ Successful: {successful_downloads}/{total_downloads}")
    print(f"✗ Failed: {total_downloads - successful_downloads}/{total_downloads}")

    if successful_downloads == total_downloads:
        print("\n🎉 All data files are ready!")
    else:
        print(
            f"\n⚠️  Warning: {total_downloads - successful_downloads} file(s) missing or failed"
        )

    print(f"\n✅ Data loading process completed!")
    print(f"📁 All files location: {data_dir}\n")


if __name__ == "__main__":
    main()
