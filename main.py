"""
Gold Price Forecasting - Main Pipeline
Orchestrates the complete end-to-end machine learning workflow
Author: Kevin Murengezi
Date: 2025
"""

import sys
from pathlib import Path
from datetime import datetime

# Configure sys.path to import from src directory
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def print_header(title):
    """Print a formatted header for each pipeline stage."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_success(message):
    """Print a success message in green (cross-platform)."""
    print(f"\n✅ {message}\n")


def print_error(message):
    """Print an error message in red (cross-platform)."""
    print(f"\n❌ ERROR: {message}\n")


def print_separator():
    """Print a visual separator."""
    print("\n" + "-" * 80 + "\n")


def main():
    """
    Main pipeline orchestrator.
    Executes the complete ML workflow in sequence:
    1. Data Loading
    2. Feature Engineering
    3. Model Training
    """

    # Pipeline start
    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("  🚀 GOLD PRICE FORECASTING - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"\n📅 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project root: {project_root}")

    # Track pipeline status
    pipeline_status = {
        "data_loader": False,
        "feature_engineering": False,
        "models": False,
    }

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    try:
        print_header("STEP 1/3: DATA LOADING")
        print("📥 Downloading historical financial data from Yahoo Finance and FRED...")

        # Import and execute data_loader
        import data_loader

        data_loader.main()

        pipeline_status["data_loader"] = True
        print_success("Data loading completed successfully!")

    except ImportError as e:
        print_error(f"Failed to import data_loader module: {e}")
        print("   Please ensure 'src/data_loader.py' exists and is properly formatted.")
        sys.exit(1)

    except Exception as e:
        print_error(f"Data loading failed: {e}")
        print("   Please check your internet connection and API access.")
        sys.exit(1)

    print_separator()

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    try:
        print_header("STEP 2/3: FEATURE ENGINEERING")
        print("🔧 Processing raw data and creating features...")

        # Import and execute feature_engineering
        import feature_engineering

        feature_engineering.main()

        pipeline_status["feature_engineering"] = True
        print_success("Feature engineering completed successfully!")

    except ImportError as e:
        print_error(f"Failed to import feature_engineering module: {e}")
        print(
            "   Please ensure 'src/feature_engineering.py' exists and is properly formatted."
        )
        sys.exit(1)

    except FileNotFoundError as e:
        print_error(f"Required data files not found: {e}")
        print("   Please ensure Step 1 (Data Loading) completed successfully.")
        sys.exit(1)

    except Exception as e:
        print_error(f"Feature engineering failed: {e}")
        print("   Please check the raw data files and feature engineering logic.")
        sys.exit(1)

    print_separator()

    # =========================================================================
    # STEP 3: MODEL TRAINING
    # =========================================================================
    try:
        print_header("STEP 3/3: MODEL TRAINING & EVALUATION")
        print("🤖 Training Random Forest model with time series cross-validation...")

        # Import and execute models
        import models

        models.main()

        pipeline_status["models"] = True
        print_success("Model training completed successfully!")

    except ImportError as e:
        print_error(f"Failed to import models module: {e}")
        print("   Please ensure 'src/models.py' exists and is properly formatted.")
        sys.exit(1)

    except FileNotFoundError as e:
        print_error(f"Required dataset not found: {e}")
        print("   Please ensure Step 2 (Feature Engineering) completed successfully.")
        sys.exit(1)

    except Exception as e:
        print_error(f"Model training failed: {e}")
        print("   Please check the processed dataset and model configuration.")
        sys.exit(1)

    # =========================================================================
    # PIPELINE COMPLETION
    # =========================================================================

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("  🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # Status summary
    print("\n📊 Pipeline Status:")
    print(
        f"  ✓ Step 1: Data Loading        - {'✅ SUCCESS' if pipeline_status['data_loader'] else '❌ FAILED'}"
    )
    print(
        f"  ✓ Step 2: Feature Engineering - {'✅ SUCCESS' if pipeline_status['feature_engineering'] else '❌ FAILED'}"
    )
    print(
        f"  ✓ Step 3: Model Training      - {'✅ SUCCESS' if pipeline_status['models'] else '❌ FAILED'}"
    )

    # Time summary
    print(f"\n⏱️  Execution Time:")
    print(f"  Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"  Duration: {duration.total_seconds():.2f} seconds ({duration.total_seconds()/60:.2f} minutes)"
    )

    # Output files
    print(f"\n📁 Output Files Generated:")
    print(f"  📂 data/raw/              - Raw financial data (CSV files)")
    print(f"  📂 data/processed/        - Processed dataset (dataset_final.csv)")
    print(f"  📂 models/                - Trained model (random_forest_model.joblib)")
    print(
        f"  📂 results/               - Plots and metrics (feature_importance.png, metrics.txt)"
    )

    # Next steps
    print(f"\n🚀 Next Steps:")
    print(f"  • Review model performance in: results/metrics.txt")
    print(f"  • Check feature importance in: results/feature_importance.png")
    print(
        f"  • Use the trained model for predictions: models/random_forest_model.joblib"
    )
    print(f"  • Consider backtesting or deploying the model!")

    print("\n" + "=" * 80)
    print("  ✨ Thank you for using the Gold Price Forecasting Pipeline!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        print("   Exiting gracefully...\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error in pipeline: {e}")
        print("   Please check all modules and try again.")
        sys.exit(1)
