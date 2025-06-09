# Stock Movement Predictor 📈

A web application built with Streamlit that uses a machine learning model to predict the next day's price movement (Up or Down) for a given stock ticker. The app also provides a historical performance review of the model's recent predictions.

**Live Demo:** [[**[Link to your deployed Streamlit App]**]([https://your-app-url.streamlit.app/](https://practicestockpredictor.streamlit.app/))](https://practicestockpredictor.streamlit.app/) 

![image](https://github.com/user-attachments/assets/ceb0f17b-349f-4a18-8a21-d2219dc4bd6a)

---

## 📋 Table of Contents
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)

---

## ✨ Features

- **Next-Day Prediction:** Enter any valid stock ticker to get a prediction for the next trading day's price movement (▲ UP or ▼ DOWN) along with the model's confidence.
- **Historical Performance Review:** View a table of the model's performance over the last 10 days, showing the predicted movement, the actual outcome, and whether the prediction was correct.
- **Data Visualization:** See a line chart of the stock's recent closing price along with its 20-day and 50-day moving averages.
- **Dynamic User Interface:** The app is fully interactive, built with a clean and modern UI using Streamlit.
- **On-Demand Model Training:** (Local Demo Only) An advanced option in the sidebar allows for re-training the model on the latest 5-year data for a specific stock.

---

## 🛠️ Technologies Used

- **Python:** Core programming language.
- **Streamlit:** For building the interactive web application UI.
- **Scikit-learn:** For training and using the `RandomForestClassifier` machine learning model.
- **Pandas:** For data manipulation, feature engineering, and analysis.
- **yfinance:** To download historical stock market data from Yahoo! Finance.
- **Joblib:** For saving and loading the trained Scikit-learn model.
- **NumPy:** For numerical operations.

---

## 📂 Project Structure

The project is organized to separate the core logic from the application interface, following best practices.
