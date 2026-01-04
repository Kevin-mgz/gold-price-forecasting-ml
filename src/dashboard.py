"""
Gold Price Forecasting - Interactive Dashboard
Professional web interface for AI-powered gold price predictions
Author: Kevin Murengezi
Date: 2025

Usage: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Gold Price AI Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .disclaimer-box {
        background-color: #FFF8E1;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin-top: 2rem;
        border-radius: 5px;
        color: #333333;
    }
    .disclaimer-box h4 {
        color: #B71C1C;
    }
    .disclaimer-box ul, .disclaimer-box p {
        color: #333333;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# DATA LOADING FUNCTIONS (WITH CACHING)
# ============================================================================


@st.cache_resource
def load_model():
    """
    Load the trained Random Forest model with caching.

    Returns:
        model: Loaded scikit-learn model
    """
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        model_path = project_root / "models" / "random_forest_model.joblib"

        if not model_path.exists():
            st.error(f"❌ Model not found at: {model_path}")
            st.info("Please run 'python src/models.py' first to train the model.")
            st.stop()

        model = joblib.load(model_path)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_data():
    """
    Load the processed dataset with caching.

    Returns:
        DataFrame: Processed dataset with date index
    """
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_path = project_root / "data" / "processed" / "dataset_final.csv"

        if not data_path.exists():
            st.error(f"❌ Dataset not found at: {data_path}")
            st.info("Please run 'python src/feature_engineering.py' first.")
            st.stop()

        # Load with date as index
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


# ============================================================================
# PREDICTION LOGIC
# ============================================================================


def prepare_features(df):
    """
    Prepare features from the latest data point.

    Args:
        df (DataFrame): Full dataset

    Returns:
        tuple: (features_df, latest_date, gold_price)
    """
    # Get the latest row
    latest_row = df.iloc[-1]
    latest_date = df.index[-1]

    # Extract gold price if available
    gold_price = df["Gold_Close"].iloc[-1] if "Gold_Close" in df.columns else None

    # Define columns to exclude
    exclude_columns = [
        "Target",
        "Gold_Close",
        "DXY_Close",
        "Rates_10Y",
        "Real_Rates_10Y",
    ]

    # Get numeric feature columns only
    feature_columns = [
        col
        for col in df.columns
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Extract features for prediction
    features = latest_row[feature_columns].to_frame().T

    return features, latest_date, gold_price


def make_prediction(model, features):
    """
    Generate prediction and confidence score.

    Args:
        model: Trained model
        features (DataFrame): Feature values

    Returns:
        tuple: (prediction, confidence, probabilities)
    """
    # Make prediction
    prediction = model.predict(features)[0]

    # Get probabilities
    probabilities = model.predict_proba(features)[0]

    # Confidence is the probability of predicted class
    confidence = probabilities[prediction]

    return prediction, confidence, probabilities


def calculate_risk_metrics(df):
    """
    Calcule les métriques de surveillance et de risque.
    """
    # 1. Détection d'Anomalies (Z-Score sur les rendements)
    # Si le Z-Score > 3 (ou < -3), c'est un mouvement anormal (3 écarts-types)
    df["Returns"] = df["Gold_Close"].pct_change()
    mean_return = df["Returns"].mean()
    std_return = df["Returns"].std()

    # Z-Score actuel (dernière semaine)
    current_return = df["Returns"].iloc[-1]
    z_score = (current_return - mean_return) / std_return

    is_anomaly = abs(z_score) > 2.5  # Seuil d'alerte (ex: 2.5 sigma)

    # 2. Value at Risk (VaR) Historique (95% et 99%)
    # "Combien je risque de perdre au pire cas (dans 95% des semaines normales) ?"
    var_95 = df["Returns"].quantile(0.05) * 100
    var_99 = df["Returns"].quantile(0.01) * 100

    return z_score, is_anomaly, var_95, var_99


# ============================================================================
# UI COMPONENTS
# ============================================================================


def display_sidebar():
    st.sidebar.title("🤖 Model Monitor")

    # === AUTOMATION: DYNAMIC LOADING ===
    # Default values (fallback in case the file doesn't exist yet)
    metrics = {
        "Accuracy": "Calculating...",
        "Win Rate": "Calculating...",
        "Total Return": "Calculating...",
        "Sharpe Ratio": "Calculating...",
    }

    # Attempt to load the file generated by evaluation.py

    try:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> racine du repo
        metrics_path = PROJECT_ROOT / "results" / "latest_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                loaded_metrics = json.load(f)
                # Update with actual values
                metrics.update(loaded_metrics)
        else:
            st.sidebar.warning(f"File not found: {metrics_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading metrics: {e}")

    # ==============================================
    def fmt2(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else x

    # Display metrics
    st.sidebar.markdown("### 📊 Backtest Performance")
    st.sidebar.markdown("*(On unseen test data)*")

    # 1. Accuracy
    st.sidebar.metric("Accuracy", fmt2(metrics["Accuracy"]))
    st.sidebar.caption(
        "🎯 **Prediction Quality:** How often the AI correctly predicted the weekly direction (Up/Down)."
    )

    # 3. Total Return
    st.sidebar.metric("Total Return", fmt2(metrics["Total Return"]))
    st.sidebar.caption(
        "💰 **Net Profit:** Cumulative gain generated by the strategy during the test period."
    )

    # 2. Win Rate
    st.sidebar.metric("Win Rate", fmt2(metrics["Win Rate"]))
    st.sidebar.caption(
        "✅ **Trading Success:** Percentage of trades that actually resulted in a profit."
    )

    # 4. Sharpe Ratio
    st.sidebar.metric("Sharpe Ratio", fmt2(metrics["Sharpe Ratio"]))
    st.sidebar.caption(
        "⚖️ **Risk-Adjusted Return:** Measure of performance relative to risk taken. > 2.0 is excellent."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Model:** Random Forest Classifier\n\n"
        "**Training Period:** 2005-2021\n\n"
        "**Testing Period:** 2021-2025 \n\n"
        "**Features:** 10+ technical & fundamental indicators"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 🎯 How it works
        
        The AI analyzes:
        - 📈 Price momentum & trends
        - 💵 Dollar strength (DXY)
        - 📊 Interest rates
        - 🏛️ Fed policy indicators
        - 📉 Technical indicators
    """
    )


def display_market_snapshot(latest_date, gold_price, df):
    """
    Display market snapshot with key metrics.

    Args:
        latest_date (datetime): Latest data date
        gold_price (float): Current gold price
        df (DataFrame): Full dataset
    """
    st.markdown("### 📸 Market Snapshot")

    col1, col2, col3 = st.columns(3)

    # Column 1: Current Date
    with col1:
        st.metric(
            label="📅 Latest Data",
            value=latest_date.strftime("%Y-%m-%d"),
            delta=latest_date.strftime("%A"),
        )

    # Column 2: Current Gold Price
    with col2:
        if gold_price is not None:
            # Calculate 1-day change
            prev_price = df["Gold_Close"].iloc[-2] if len(df) > 1 else gold_price
            price_change = gold_price - prev_price
            price_change_pct = (price_change / prev_price) * 100

            st.metric(
                label="💰 Gold Price",
                value=f"${gold_price:.2f}",
                delta=f"{price_change_pct:+.2f}%",
            )

    # Column 3: 4-Week Trend
    with col3:
        if "Gold_Close" in df.columns and len(df) >= 20:
            # Calculate 4-week (20 trading days) change
            price_4w_ago = df["Gold_Close"].iloc[-20]
            trend_change = ((gold_price / price_4w_ago) - 1) * 100

            st.metric(
                label="📊 4-Week Trend",
                value=f"{trend_change:+.2f}%",
                delta="Uptrend" if trend_change > 0 else "Downtrend",
            )


def display_market_snapshot(latest_date, gold_price, df):
    """
    Display market snapshot with key metrics.

    Args:
        latest_date (datetime): Latest data date
        gold_price (float): Current gold price
        df (DataFrame): Full dataset
    """
    st.markdown("### 📸 Market Snapshot")

    col1, col2, col3 = st.columns(3)

    # Column 1: Current Date
    with col1:
        st.metric(
            label="📅 Latest Data",
            value=latest_date.strftime("%Y-%m-%d"),
            delta=latest_date.strftime("%A"),
        )

    # Column 2: Current Gold Price
    with col2:
        if gold_price is not None:
            # Calculate 1-day change
            prev_price = df["Gold_Close"].iloc[-2] if len(df) > 1 else gold_price
            price_change = gold_price - prev_price
            price_change_pct = (price_change / prev_price) * 100

            st.metric(
                label="💰 Gold Price",
                value=f"${gold_price:.2f}",
                delta=f"{price_change_pct:+.2f}%",
            )

    # Column 3: 4-Week Trend
    with col3:
        if "Gold_Close" in df.columns and len(df) >= 20:
            # Calculate 4-week (20 trading days) change
            price_4w_ago = df["Gold_Close"].iloc[-20]
            trend_change = ((gold_price / price_4w_ago) - 1) * 100

            st.metric(
                label="📊 4-Week Trend",
                value=f"{trend_change:+.2f}%",
                delta="Uptrend" if trend_change > 0 else "Downtrend",
            )


def display_prediction(prediction, confidence, probabilities):
    """
    Display AI prediction with visual emphasis.

    Args:
        prediction (int): 0 (Down) or 1 (Up)
        confidence (float): Confidence score (0-1)
        probabilities (array): Probability for each class
    """
    st.markdown("### 🤖 AI Prediction")

    # Determine signal details
    if prediction == 1:
        signal_text = "🚀 SIGNAL: ACHAT (HAUSSE PRÉVUE)"
        signal_type = "success"
        interpretation = (
            "Le modèle détecte des conditions favorables à une hausse du prix de l'or. "
            "Les indicateurs techniques et fondamentaux suggèrent un momentum positif."
        )
        color = "#28a745"
    else:
        signal_text = "🔻 SIGNAL: VENTE / CASH (BAISSE PRÉVUE)"
        signal_type = "error"
        interpretation = (
            "Le modèle anticipe une correction ou consolidation du prix de l'or. "
            "Les conditions du marché suggèrent une pression baissière à court terme."
        )
        color = "#dc3545"

    # Display main signal
    if signal_type == "success":
        st.success(signal_text, icon="🚀")
    else:
        st.error(signal_text, icon="🔻")

    # Display confidence
    st.markdown("#### 📊 Confidence Score")

    # Confidence interpretation
    if confidence >= 0.75:
        conf_text = "🟢 Very High Confidence"
        conf_color = "green"
    elif confidence >= 0.65:
        conf_text = "🟡 High Confidence"
        conf_color = "orange"
    elif confidence >= 0.55:
        conf_text = "🟠 Moderate Confidence"
        conf_color = "orange"
    else:
        conf_text = "🔴 Low Confidence"
        conf_color = "red"

    # Progress bar for confidence
    st.progress(confidence, text=f"{conf_text}: {confidence*100:.1f}%")

    # Detailed probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📉 Probability Down", f"{probabilities[0]*100:.1f}%")
    with col2:
        st.metric("📈 Probability Up", f"{probabilities[1]*100:.1f}%")

    # Interpretation
    st.info(f"💡 **Interpretation:** {interpretation}")


def display_historical_chart(df):
    """
    Display interactive historical price chart.

    Args:
        df (DataFrame): Full dataset (weekly data)
    """
    st.markdown("### 📈 Historical Context")

    if "Gold_Close" not in df.columns:
        st.warning("Gold price data not available for chart.")
        return

    # STEP 1: Calculate 50-week MA on FULL dataset first
    # (This ensures we have enough historical data for accurate MA calculation)
    df_with_ma = df.copy()
    df_with_ma["MA_50"] = df_with_ma["Gold_Close"].rolling(window=50).mean()

    # STEP 2: NOW filter to last 52 weeks (1 year of weekly data)
    lookback_weeks = 52
    df_recent = df_with_ma.tail(lookback_weeks)

    # Create interactive plotly chart
    fig = go.Figure()

    # Add gold price line
    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=df_recent["Gold_Close"],
            mode="lines",
            name="Gold Price",
            line=dict(color="#FFD700", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 215, 0, 0.1)",
        )
    )

    # Add 50-week moving average (if we have valid values)
    if df_recent["MA_50"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df_recent.index,
                y=df_recent["MA_50"],
                mode="lines",
                name="50-Week MA",
                line=dict(color="#FF6B6B", width=1, dash="dash"),
            )
        )

    # Customize layout
    fig.update_layout(
        title="Gold Price Trend (Last 1 Year - Weekly Data)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_disclaimer():
    """Display legal disclaimer at bottom of page."""
    st.markdown("---")
    st.markdown(
        """
        <div class="disclaimer-box">
            <h4>⚠️ IMPORTANT DISCLAIMER</h4>
            <p>
                <strong>This application is for EDUCATIONAL PURPOSES ONLY.</strong>
            </p>
            <ul>
                <li>❌ This is NOT financial advice or investment recommendation</li>
                <li>❌ Past performance does not guarantee future results</li>
                <li>❌ AI predictions can be wrong and should not be relied upon</li>
                <li>✅ Always consult with a qualified financial advisor</li>
                <li>✅ Never invest more than you can afford to lose</li>
            </ul>
            <p>
                The creators of this application are not responsible for any financial 
                losses incurred from using this tool. Trading and investing involve 
                substantial risk of loss.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main dashboard application."""

    # Display sidebar
    display_sidebar()

    # Main header
    st.markdown(
        '<h1 class="main-header">🏆 Gold Price AI Advisor</h1>', unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666;'>"
        "AI-Powered Trading Signals for Gold Investors"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Load model and data
    with st.spinner("🔄 Loading AI model and market data..."):
        model = load_model()
        df = load_data()

    # Prepare features and make prediction
    features, latest_date, gold_price = prepare_features(df)
    prediction, confidence, probabilities = make_prediction(model, features)

    # Display market snapshot
    display_market_snapshot(latest_date, gold_price, df)

    st.markdown("---")

    # Display prediction
    display_prediction(prediction, confidence, probabilities)

    st.markdown("---")

    # === RISK & SURVEILLANCE SECTION (MODIFIED) ===
    st.markdown("### 🛡️ Risk & Surveillance Monitor")

    # Calculate metrics
    z_score, is_anomaly, var_95, var_99 = calculate_risk_metrics(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Volatility Z-Score", f"{z_score:.2f}")
        if is_anomaly:
            st.error("⚠️ ANOMALY DETECTED")
            st.caption(
                "🚨 **Interpretation:** Rare and significant movement (> 2.5 standard deviations). Likely driven by major news. High vigilance required."
            )
        else:
            st.success("✅ Normal Market Behavior")
            st.caption(
                "ℹ️ **Interpretation:** Current volatility is within historical norms. Standard market noise."
            )

    with col2:
        st.metric("VaR (95%)", f"{var_95:.2f}%", help="Value at Risk 95%")
        st.caption(
            f"📉 **Standard Risk:** In 95% of normal weeks, losses won't exceed **{var_95:.2f}%**. This is your standard downside protection level."
        )

    with col3:
        st.metric("VaR (99%)", f"{var_99:.2f}%", help="Value at Risk 99%")
        st.caption(
            f"💥 **Extreme Risk:** In the worst 1% of cases (Crash), losses could reach **{var_99:.2f}%**. This is your stress test limit."
        )

    st.markdown("---")
    # ==============================================

    # Display historical chart
    display_historical_chart(df)

    # Display disclaimer
    display_disclaimer()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #999;'>"
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Powered by Random Forest ML Model"
        "</p>",
        unsafe_allow_html=True,
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
