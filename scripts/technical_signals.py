"""
This script contains the original technical indicator functions and a function to store the computed signals in a database.

The apply_trading_strategy function applies the technical indicator functions to compute trading signals for a given DataFrame of price data.
The update_technical_signals function loads raw price data from the database, applies the trading strategy, and stores the computed technical signals in the 'technical_signals' table.

Technical indicators included:
- Simple Moving Average (SMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- Moving Average Convergence Divergence (MACD)
- On-Balance Volume (OBV)
- Stochastic Oscillator
- Average Directional Index (ADX)

The overall signal is computed based on a threshold from the combined indicator signals.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import (
    DB_PATH, LOG_DIR, LOG_FILE, LOG_LEVEL,
    SMA_SHORT_WINDOW, SMA_LONG_WINDOW,
    RSI_WINDOW,
    BB_WINDOW, BB_MULTIPLIER,
    MACD_SHORT_SPAN, MACD_LONG_SPAN, MACD_SIGNAL_SPAN,
    OBV_MA_WINDOW,
    STOCH_WINDOW, STOCH_SMOOTH_WINDOW,
    ADX_WINDOW,
    TECH_SIGNAL_THRESHOLD
)

# Configure logging using centralized config values
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#########################
# TECHNICAL INDICATOR FUNCTIONS
#########################

def compute_sma_signals(df, short_window=SMA_SHORT_WINDOW, long_window=SMA_LONG_WINDOW):
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    df['SMA_Ratio'] = df['SMA_long'] / df['SMA_short']
    df['SMA_signal'] = np.where(df['SMA_Ratio'] < 1, 1, 0)
    df['SMA_position'] = df['SMA_signal'].diff()
    return df

def compute_rsi(df, window=RSI_WINDOW):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta).clip(lower=0).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_signal'] = np.where(df['RSI'] < 30, 1,
                                np.where(df['RSI'] > 70, -1, 0))
    return df

def compute_bollinger_bands(df, window=BB_WINDOW):
    df['BB_MA'] = df['close'].rolling(window=window).mean()
    df['BB_std'] = df['close'].rolling(window=window).std()
    df['BB_upper'] = df['BB_MA'] + BB_MULTIPLIER * df['BB_std']
    df['BB_lower'] = df['BB_MA'] - BB_MULTIPLIER * df['BB_std']
    df['BB_signal'] = np.where(df['close'] < df['BB_lower'], 1,
                               np.where(df['close'] > df['BB_upper'], -1, 0))
    return df

def compute_macd(df, span_short=MACD_SHORT_SPAN, span_long=MACD_LONG_SPAN, span_signal=MACD_SIGNAL_SPAN):
    df['EMA_short'] = df['close'].ewm(span=span_short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=span_long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal_line'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()
    df['MACD_signal'] = np.where(df['MACD'] > df['MACD_signal_line'], 1, -1)
    return df

def compute_obv(df, ma_window=OBV_MA_WINDOW):
    df['price_change'] = df['close'].diff()
    df['direction'] = np.where(df['price_change'] > 0, 1,
                               np.where(df['price_change'] < 0, -1, 0))
    df['OBV'] = (df['volume'] * df['direction']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=ma_window).mean()
    df['OBV_signal'] = np.where(df['OBV'] > df['OBV_MA'], 1, -1)
    return df

def compute_stochastic(df, window=STOCH_WINDOW, smooth_window=STOCH_SMOOTH_WINDOW):
    df['lowest_low'] = df['close'].rolling(window=window).min()
    df['highest_high'] = df['close'].rolling(window=window).max()
    df['%K'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['%D'] = df['%K'].rolling(window=smooth_window).mean()
    df['stoch_signal'] = np.where(df['%K'] < 20, 1,
                                  np.where(df['%K'] > 80, -1, 0))
    return df

def compute_adx(df, window=ADX_WINDOW):
    df['TR'] = df['close'].diff().abs()
    df['+DM'] = np.where(df['close'].diff() > 0, df['close'].diff(), 0)
    df['-DM'] = np.where(df['close'].diff() < 0, -df['close'].diff(), 0)
    df['TR_smooth'] = df['TR'].rolling(window=window).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=window).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=window).sum()
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    df['DX'] = 100 * (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=window).mean()
    df['ADX_signal'] = np.where((df['ADX'] > 25) & (df['+DI'] > df['-DI']), 1,
                                np.where((df['ADX'] > 25) & (df['+DI'] < df['-DI']), -1, 0))
    return df

def apply_trading_strategy(df):
    """
    Applies all technical indicator functions and combines their signals.
    """
    df = df.sort_values(by='date').reset_index(drop=True)
    df = compute_sma_signals(df)
    df = compute_rsi(df)
    df = compute_bollinger_bands(df)
    df = compute_macd(df)
    df = compute_obv(df)
    df = compute_stochastic(df)
    df = compute_adx(df)
    
    # Combine signals from 7 indicators:
    signal_cols = ['SMA_signal', 'RSI_signal', 'BB_signal', 
                   'MACD_signal', 'OBV_signal', 'stoch_signal', 'ADX_signal']
    for col in signal_cols:
        if col not in df.columns:
            df[col] = 0
    df['combined_signal_score'] = df[signal_cols].sum(axis=1)
    df['overall_signal'] = np.where(df['combined_signal_score'] >= TECH_SIGNAL_THRESHOLD, 1,
                                    np.where(df['combined_signal_score'] <= -TECH_SIGNAL_THRESHOLD, -1, 0))
    # Compute ATR as a volatility proxy using a 14-day window (consistent with RSI_WINDOW, can be adjusted if needed)
    df['atr'] = df['close'].diff().abs().rolling(window=14).mean()
    return df

#########################
# SQL INTEGRATION & TECHNICAL SIGNAL STORAGE
#########################

def update_technical_signals(db_path):
    """
    Loads raw price data from the database, applies the trading strategy, and
    stores the computed technical signals in the 'technical_signals' table.
    """
    try:
        conn = sqlite3.connect(db_path)
        logging.info("Connected to database for technical signals update.")
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return
    
    # Retrieve daily prices
    price_query = "SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices"
    price_df = pd.read_sql_query(price_query, conn, parse_dates=['date'])
    logging.info("Loaded daily price data from nasdaq_100_daily_prices.")
    
    technical_records = []
    
    for ticker, group in price_df.groupby('ticker'):
        group = group.sort_values(by='date').reset_index(drop=True)
        group = apply_trading_strategy(group)
        # Drop rows where key indicators are missing (for example, SMA_long, RSI, MACD, atr)
        group = group.dropna(subset=['SMA_long', 'RSI', 'MACD', 'atr'])
        for _, row in group.iterrows():
            record = (
                ticker,
                row['date'].strftime('%Y-%m-%d'),
                row['SMA_long'],  # Using 20-day SMA as an example indicator
                row['RSI'],
                row['MACD'],
                row['atr'],
                int(row['overall_signal'])
            )
            technical_records.append(record)
    
    insert_query = """
        INSERT OR IGNORE INTO technical_signals 
        (ticker, signal_date, sma, rsi, macd, atr, signal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    try:
        cursor = conn.cursor()
        cursor.executemany(insert_query, technical_records)
        conn.commit()
        logging.info(f"Inserted {len(technical_records)} records into technical_signals.")
        print(f"Inserted {len(technical_records)} records into technical_signals.")
    except Exception as e:
        logging.error(f"Error inserting technical signals: {e}")
    finally:
        conn.close()

# Allow module testing
if __name__ == "__main__":
    update_technical_signals(DB_PATH)
