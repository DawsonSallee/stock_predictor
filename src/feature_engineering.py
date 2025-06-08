import pandas as pd

def create_features(df):
    """Creates time-series features from a stock data DataFrame."""
    df_new = df.copy()
    # Create moving averages
    df_new['ma20'] = df_new['Close'].rolling(window=20).mean()
    df_new['ma50'] = df_new['Close'].rolling(window=50).mean()
    # Create volatility
    df_new['volatility'] = df_new['Close'].rolling(window=20).std()
    # Create Relative Strength Index (RSI)
    delta = df_new['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_new['rsi'] = 100 - (100 / (1 + rs))

    df_new.dropna(inplace=True) # Remove rows with NaN values created by rolling windows
    return df_new
