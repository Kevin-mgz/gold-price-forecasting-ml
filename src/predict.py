"""
Gold Price Forecasting - Prediction Advisor Script
Provides trading signals based on the trained model
Author: Murengezi Kevin
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def print_header():
    """Print formatted header."""
    print("\n" + "=" * 80)
    print("  💰 GOLD PRICE PREDICTION ADVISOR")
    print("=" * 80 + "\n")


def print_separator():
    """Print visual separator."""
    print("-" * 80)


def setup_paths():
    """
    Setup paths for model and data files.

    Returns:
        tuple: (model_path, data_path)
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    model_path = project_root / "models" / "random_forest_model.joblib"
    data_path = project_root / "data" / "processed" / "dataset_final.csv"

    return model_path, data_path


def load_model(model_path):
    """
    Load the trained Random Forest model.

    Args:
        model_path (Path): Path to the saved model

    Returns:
        model: Loaded scikit-learn model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ Model not found at: {model_path}\n"
            f"   Please run 'python src/models.py' first to train the model.\n"
        )

    print(f"📦 Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully\n")

    return model


def load_latest_data(data_path):
    """
    Load the dataset and extract the latest market data.

    Args:
        data_path (Path): Path to the processed dataset

    Returns:
        tuple: (latest_features, latest_date, gold_price, full_df)

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {data_path}\n"
            f"   Please run 'python src/feature_engineering.py' first.\n"
        )

    print(f"📂 Loading data from: {data_path}")

    # Load with date as index
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Get the latest row (most recent market data)
    latest_row = df.iloc[-1]
    latest_date = df.index[-1]

    # Extract current gold price if available
    if "Gold_Close" in df.columns:
        gold_price = df["Gold_Close"].iloc[-1]
    else:
        gold_price = None

    # Define columns to exclude from features
    exclude_columns = [
        "Target",
        "Gold_Close",
        "DXY_Close",
        "Rates_10Y",
        "Real_Rates_10Y",
    ]

    # Get feature columns (only numeric, excluding reference columns)
    feature_columns = [
        col
        for col in df.columns
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Extract features for prediction
    latest_features = latest_row[feature_columns].to_frame().T

    print(f"📊 Latest market data:")
    print(f"  Date:           {latest_date.strftime('%Y-%m-%d (%A)')}")
    if gold_price is not None:
        print(f"  Gold Price:     ${gold_price:.2f}")
    print(f"  Features used:  {len(feature_columns)} indicators")

    return latest_features, latest_date, gold_price, df


def make_prediction(model, features):
    """
    Make prediction and get confidence probability.

    Args:
        model: Trained model
        features (DataFrame): Feature values for prediction

    Returns:
        tuple: (prediction, confidence)
    """
    print(f"\n🤖 Analyzing market conditions...")

    # Make prediction (0 or 1)
    prediction = model.predict(features)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(features)[0]

    # Confidence is the probability of the predicted class
    confidence = probabilities[prediction]

    return prediction, confidence


def display_signal(prediction, confidence, latest_date, gold_price):
    """
    Display the trading signal with visual formatting.

    Args:
        prediction (int): 0 (Down) or 1 (Up)
        confidence (float): Confidence probability (0-1)
        latest_date (datetime): Date of latest data
        gold_price (float): Current gold price
    """
    print("\n" + "=" * 80)
    print("  🎯 TRADING SIGNAL")
    print("=" * 80 + "\n")

    # Display signal based on prediction
    if prediction == 1:
        signal_emoji = "🚀"
        signal_text = "ACHAT (HAUSSE PRÉVUE)"
        signal_color = "UP ↗"
        recommendation = "Envisagez d'acheter ou de conserver vos positions en or."
        risk_note = "Le modèle anticipe une hausse du prix de l'or."
    else:
        signal_emoji = "🔻"
        signal_text = "VENTE / CASH (BAISSE PRÉVUE)"
        signal_color = "DOWN ↘"
        recommendation = "Envisagez de vendre ou de rester en liquidités (cash)."
        risk_note = "Le modèle anticipe une baisse du prix de l'or."

    # Display main signal
    print(f"{signal_emoji} SIGNAL: {signal_text}")
    print(f"\n  Direction:      {signal_color}")
    print(f"  Confiance:      {confidence*100:.1f}%")

    # Confidence level interpretation
    if confidence >= 0.80:
        conf_level = "🟢 TRÈS HAUTE"
        conf_note = "Le modèle est très confiant dans cette prédiction."
    elif confidence >= 0.65:
        conf_level = "🟡 HAUTE"
        conf_note = "Le modèle est assez confiant dans cette prédiction."
    elif confidence >= 0.55:
        conf_level = "🟠 MOYENNE"
        conf_note = "Le modèle est modérément confiant. Prudence recommandée."
    else:
        conf_level = "🔴 FAIBLE"
        conf_note = "Signal incertain. Attendre une meilleure opportunité."

    print(f"  Niveau:         {conf_level}")
    print(f"\n📋 Interprétation:")
    print(f"  {risk_note}")
    print(f"  {conf_note}")
    print(f"\n💡 Recommandation:")
    print(f"  {recommendation}")

    # Add confidence bar
    print(f"\n📊 Confidence Score:")
    bar_length = 50
    filled_length = int(bar_length * confidence)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"  [{bar}] {confidence*100:.1f}%")


def display_disclaimer():
    """Display legal disclaimer."""
    print("\n" + "=" * 80)
    print("  ⚠️  AVERTISSEMENT IMPORTANT")
    print("=" * 80)
    print(
        """
⚠️  DISCLAIMER - À LIRE ATTENTIVEMENT:

    Ce système est un OUTIL ÉDUCATIF basé sur l'apprentissage automatique.
    
    ❌ CECI N'EST PAS UN CONSEIL FINANCIER
    ❌ CECI N'EST PAS UNE RECOMMANDATION D'INVESTISSEMENT
    
    • Les prédictions sont basées sur des données historiques et peuvent être
      incorrectes.
    • Les marchés financiers sont imprévisibles et comportent des risques
      importants.
    • Vous êtes seul responsable de vos décisions d'investissement.
    • Consultez toujours un conseiller financier professionnel avant d'investir.
    • Ne jamais investir plus que ce que vous pouvez vous permettre de perdre.
    
    Les performances passées ne garantissent pas les résultats futurs.
"""
    )
    print("=" * 80 + "\n")


def display_additional_info(df, latest_date):
    """
    Display additional market context and historical performance.

    Args:
        df (DataFrame): Full dataset
        latest_date (datetime): Latest data date
    """
    print("\n" + "=" * 80)
    print("  📈 CONTEXTE ADDITIONNEL")
    print("=" * 80 + "\n")

    # Recent price trend
    if "Gold_Close" in df.columns:
        recent_prices = df["Gold_Close"].tail(30)
        price_change_30d = ((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1) * 100

        print(f"📊 Performance récente (30 derniers jours):")
        print(f"  Variation:      {price_change_30d:+.2f}%")
        print(f"  Prix min:       ${recent_prices.min():.2f}")
        print(f"  Prix max:       ${recent_prices.max():.2f}")
        print(f"  Prix actuel:    ${recent_prices.iloc[-1]:.2f}")

    # Model info
    print(f"\n🤖 Informations sur le modèle:")
    print(f"  Algorithme:     Random Forest Classifier")
    print(f"  Données jusqu'à: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Prédiction pour: Semaine prochaine")


def main():
    """
    Main prediction pipeline.
    """
    try:
        # Display header
        print_header()

        # Setup paths
        model_path, data_path = setup_paths()

        # Load model
        model = load_model(model_path)

        # Load latest data
        latest_features, latest_date, gold_price, df = load_latest_data(data_path)

        # Make prediction
        prediction, confidence = make_prediction(model, latest_features)

        # Display trading signal
        display_signal(prediction, confidence, latest_date, gold_price)

        # Display additional context
        display_additional_info(df, latest_date)

        # Display disclaimer
        display_disclaimer()

        print("✅ Prédiction générée avec succès!")
        print(f"🕐 Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("💡 Solution: Exécutez d'abord le pipeline complet:")
        print("   1. python src/data_loader.py")
        print("   2. python src/feature_engineering.py")
        print("   3. python src/models.py")
        print("   4. python src/predict.py\n")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}\n")
        print("💡 Vérifiez que tous les fichiers nécessaires sont présents.")
        print("   Si le problème persiste, relancez le pipeline complet.\n")
        raise


if __name__ == "__main__":
    main()
