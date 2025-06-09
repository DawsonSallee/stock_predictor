import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from datetime import date, timedelta
import os
from src.feature_engineering import create_features # Import our function

# --- Page Configuration ---
st.set_page_config(page_title="Stock Movement Predictor", layout="wide")

# --- Load The Trained Model ---
MODEL_PATH = 'models/stock_predictor.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found! Please train the model first using the notebook.")
    st.stop()

# --- App Title ---
st.title("📈 Stock Movement Predictor")

# --- Sidebar for User Input ---
st.sidebar.header("User Input")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOG):", "AAPL").upper()

# --- Prediction Logic ---
st.header(f"Prediction for {stock_ticker}")

try:
    # Fetch the latest year of data for feature calculation
    data_df = yf.download(stock_ticker, period="1y")

    if not data_df.empty:
        # Engineer features for the entire DataFrame
        featured_df = create_features(data_df)

        # Get the features for the most recent day
        features_list = ['ma20', 'ma50', 'volatility', 'rsi', 'macd', 'macd_signal']
        latest_features = featured_df.iloc[[-1]][features_list]
        # Make prediction
        prediction = model.predict(latest_features)[0]
        prediction_proba = model.predict_proba(latest_features)[0]

        st.subheader(f"Next Day's Predicted Movement:")
        if prediction == 1:
            st.success(f"▲ UP (Confidence: {prediction_proba[1]:.2f})")
        else:
            st.error(f"▼ DOWN (Confidence: {prediction_proba[0]:.2f})")

        # --- Display Data and Charts ---
        st.header("Recent Data & Features")
        st.dataframe(featured_df.tail(10))

        st.header("Closing Price and Moving Averages")
        chart_data = featured_df[[('Close', stock_ticker), ('ma20', ''), ('ma50', '')]]
        chart_data.columns = ['Close', 'MA20', 'MA50']
        st.line_chart(chart_data)
    else:
        st.warning("Could not download data for the specified ticker. Please check if it's correct.")

except Exception as e:
    st.error(f"An error occurred: {e}")

# (Prediction history will be added in Phase 3)
