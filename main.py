"""
Gold Price Forecasting - Main Pipeline
Orchestrates the complete end-to-end machine learning workflow
Author: Kevin Murengezi
Date: 2025
"""

import sys
import subprocess
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


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print_header("LAUNCHING DASHBOARD")
    print("🌐 Starting Streamlit dashboard...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard.\n")

    dashboard_path = src_path / "dashboard.py"
    # Use sys.executable to ensure cross-platform compatibility (Windows/macOS/Linux)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def main():
    """
    Main pipeline orchestrator.
    Executes the complete ML workflow in sequence:
    1. Data Loading
    2. Feature Engineering
    3. Model Training
    4. Prediction
    5. (Optional) Launch Dashboard
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
        "predict": False,
    }

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    try:
        print_header("STEP 1/4: DATA LOADING")
        print("📥 Downloading historical financial data from Yahoo Finance and FRED...")

        import data_loader

        data_loader.main()

        pipeline_status["data_loader"] = True
        print_success("Data loading completed successfully!")

    except ImportError as e:
        print_error(f"Failed to import data_loader module: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        sys.exit(1)

    print_separator()

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    try:
        print_header("STEP 2/4: FEATURE ENGINEERING")
        print("🔧 Processing raw data and creating features...")

        import feature_engineering

        feature_engineering.main()

        pipeline_status["feature_engineering"] = True
        print_success("Feature engineering completed successfully!")

    except ImportError as e:
        print_error(f"Failed to import feature_engineering module: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Feature engineering failed: {e}")
        sys.exit(1)

    print_separator()

    # =========================================================================
    # STEP 3: MODEL TRAINING
    # =========================================================================
    try:
        print_header("STEP 3/4: MODEL TRAINING & EVALUATION")
        print("🤖 Training Random Forest model with time series cross-validation...")

        import models

        models.main()

        pipeline_status["models"] = True
        print_success("Model training completed successfully!")

    except ImportError as e:
        print_error(f"Failed to import models module: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Model training failed: {e}")
        sys.exit(1)

    print_separator()

    # =========================================================================
    # STEP 4: PREDICTION
    # =========================================================================
    try:
        print_header("STEP 4/4: GENERATE PREDICTION")
        print("🔮 Generating prediction for next week...")

        import predict

        predict.main()

        pipeline_status["predict"] = True
        print_success("Prediction generated successfully!")

    except ImportError as e:
        print_error(f"Failed to import predict module: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Prediction failed: {e}")
        sys.exit(1)

    # =========================================================================
    # PIPELINE COMPLETION
    # =========================================================================

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("  🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

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
    print(
        f"  ✓ Step 4: Prediction          - {'✅ SUCCESS' if pipeline_status['predict'] else '❌ FAILED'}"
    )

    print(f"\n⏱️  Duration: {duration.total_seconds():.2f} seconds")

    print(f"\n📁 Output Files Generated:")
    print(f"  • data/processed/dataset_final.csv")
    print(f"  • models/random_forest_model.joblib")
    print(f"  • results/feature_importance.png, metrics.txt")

    print(f"\n🔍 Additional Analysis:")
    print(
        f"  • Compare with Regression approach by running: python src/regression_bonus.py"
    )

    print("\n" + "=" * 80)

    # =========================================================================
    # STEP 4: DASHBOARD (OPTIONAL)
    # =========================================================================
    print("\n🌐 Would you like to launch the interactive dashboard?")

    try:
        response = input("   Launch dashboard? [y/N]: ").strip().lower()
        if response in ["y", "yes"]:
            launch_dashboard()
        else:
            print("\n💡 To launch the dashboard later, run:")
            print("   streamlit run src/dashboard.py\n")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
