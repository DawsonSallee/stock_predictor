# Stock Movement Predictor 📈

A web application built with Streamlit that uses a machine learning model to predict the next day's price movement (Up or Down) for a given stock ticker. The app also provides a historical performance review of the model's recent predictions.

**Live Demo:** [(https://practicestockpredictor.streamlit.app/))](https://practicestockpredictor.streamlit.app/)](https://practicestockpredictor.streamlit.app/) 

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

stock_predictor/
├── .gitignore
├── README.md
├── app.py # Main Streamlit application script
├── requirements.txt # Python dependencies for deployment
├── models/
│ └── stock_predictor.pkl # Pre-trained machine learning model
├── notebooks/
│ └── 1.0-model-development.ipynb # Jupyter Notebook for model experimentation and training
└── src/
└── feature_engineering.py # Module for calculating financial indicators


---

## 🧠 How It Works

### 1. Data Acquisition
The app uses the `yfinance` library to download the last year of historical data for the user-specified stock ticker.

### 2. Feature Engineering
Using the raw data, the following technical indicators are calculated to serve as features for the model:
- **Moving Averages (MA20, MA50):** Average price over the last 20 and 50 days.
- **Volatility:** Standard deviation of the price over the last 20 days.
- **Relative Strength Index (RSI):** A momentum oscillator to measure the speed and change of price movements.
- **Moving Average Convergence Divergence (MACD):** A trend-following momentum indicator.

### 3. Prediction (Inference)
- The pre-trained `RandomForestClassifier` model (`stock_predictor.pkl`) is loaded into memory.
- The features for the most recent day are fed into the model.
- The model outputs a prediction (`1` for UP, `0` for DOWN) and a confidence probability for that prediction.

### 4. Back-Testing
For the "Last 10 Days Performance Review," the application runs a historical simulation. For each day in the last 10 days, it uses the data from the *day before* to make a prediction and then compares that prediction to the actual known outcome, marking it as "Correct" or "Incorrect".

---

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/stock-predictor.git
    cd stock-predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  Enter a stock ticker in the sidebar and see the results!

---

## 🔮 Future Improvements

- **More Advanced Models:** Experiment with more complex time-series models like LSTM or Gradient Boosting Machines (XGBoost).
- **Expanded Feature Set:** Incorporate more advanced technical indicators or fundamental data.
- **User Accounts and History:** Allow users to create accounts and track the performance of their favorite stocks over time.
- **Cloud-Based Training:** Set up a scheduled cloud function (e.g., on AWS Lambda or Google Cloud Functions) to automatically re-train the model weekly and update the `.pkl` file in the repository.
