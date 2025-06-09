import streamlit as st
import pandas as pd
import numpy as np
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

def create_performance_report(df, model, features_list):
    """
    Performs a back-test on recent data to generate a performance report.

    Args:
        df (pd.DataFrame): The DataFrame with features already engineered.
        model: The trained machine learning model.
        features_list (list): The list of feature names.

    Returns:
        pd.DataFrame: A formatted DataFrame with the performance report.
    """
    # Make a copy to avoid modifying the original DataFrame
    report_df = df.copy()

    # --- Time-shifted Prediction ---
    # To predict for Day N, we must use data from Day N-1.
    # The .shift(1) method moves all feature data down by one row.
    features_for_prediction = report_df[features_list].shift(1)

    # We can only predict on rows that have historical data (i.e., not the first row)
    valid_indices = features_for_prediction.dropna().index
    valid_features = features_for_prediction.loc[valid_indices]

    # --- Batch Prediction for Efficiency ---
    # Predict on all valid historical days at once
    predictions = model.predict(valid_features)
    probabilities = model.predict_proba(valid_features)

    # Add prediction results back to the report DataFrame
    report_df.loc[valid_indices, 'Prediction'] = predictions

    # Get the confidence of the predicted class.
    # e.g., if prediction is 1, we take the probability of class 1.
    report_df.loc[valid_indices, 'Confidence'] = probabilities[np.arange(len(predictions)), predictions]

    # --- Determine Actual Outcome & Correctness ---
    # Determine the actual movement by comparing a day's close to the previous day's close
    report_df['Actual_Movement'] = np.where(report_df['Close'] > report_df['Close'].shift(1), 1, 0)

    # Compare prediction to the actual outcome
    report_df['Result'] = np.nan # Default to Not a Number
    report_df.loc[valid_indices, 'Result'] = np.where(
        report_df.loc[valid_indices, 'Prediction'] == report_df.loc[valid_indices, 'Actual_Movement'],
        '✅ Correct',
        '❌ Incorrect'
    )

    # --- Formatting for Display ---
    # Convert numeric prediction to user-friendly text
    report_df['Prediction'] = report_df['Prediction'].map({1.0: '▲ UP', 0.0: '▼ DOWN'})

    # Select and reorder columns for the final report
    report_df.reset_index(inplace=True) # Make the 'Date' index a regular column
    report_df.rename(columns={'index': 'Date'}, inplace=True)
    
    final_cols = ['Date', 'Open', 'Close', 'Prediction', 'Confidence', 'Result']

    final_report = report_df[final_cols].tail(10)

    
    return final_report.sort_values(by='Date', ascending=False)

# --- App Title ---
st.title("📈 Stock Movement Predictor")

# --- Sidebar for User Input ---
st.sidebar.header("User Input")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOG):", "VOO").upper()

# --- Prediction Logic ---
st.header(f"Analysis for {stock_ticker}")

try:
    data_df = yf.download(stock_ticker, period="1y", progress=False)

    # This standardizes the DataFrame for all downstream functions.
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = data_df.columns.droplevel(1)
    
    # st.subheader("--- DEBUGGING ---")
    # st.write("Columns received from yfinance:", data_df.columns)
    # st.dataframe(data_df.head())
    # st.stop() # This will stop the app here so we can see the output

    if not data_df.empty:
        # 3. Engineer features using the GUARANTEED CLEAN DataFrame
        featured_df = create_features(data_df)

        # --- Prediction for Tomorrow ---
        features_list = ['ma20', 'ma50', 'volatility', 'rsi', 'macd', 'macd_signal']
        latest_features = featured_df.iloc[[-1]][features_list]
        prediction = model.predict(latest_features)[0]
        prediction_proba = model.predict_proba(latest_features)[0]

        st.header(f"Next Day's Predicted Movement:")
        if prediction == 1:
            st.success(f"▲ UP (Confidence: {prediction_proba[1]:.2f})")
        else:
            st.error(f"▼ DOWN (Confidence: {prediction_proba[0]:.2f})")

        # --- Display the Historical Performance Report ---
        st.header("Last 10 Days Performance Review")

        # 4. This function will now receive clean data and will not raise a KeyError
        performance_report = create_performance_report(featured_df, model, features_list)

        st.dataframe(
            performance_report,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Open": st.column_config.NumberColumn("Open", format="$%.2f"),
                "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.2f", min_value=0, max_value=1,
                ),
            },
            hide_index=True,
        )

        # --- Display Data and Charts ---
        st.header("Recent Data & Features")
        st.dataframe(featured_df.tail(10).sort_index(ascending=False))

        st.header("Closing Price and Moving Averages")
        chart_data = featured_df[[('Close'), ('ma20'), ('ma50')]]
        chart_data.columns = ['Close', 'MA20', 'MA50']
        st.line_chart(chart_data)

    else:
        st.warning("Could not download data for the specified ticker. Please check if it's correct.")

except Exception as e:
    st.error(f"An error occurred: {e}")