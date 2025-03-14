## fetch_price.py

import sqlite3
import datetime
import logging
import pandas as pd
import yfinance as yf
from time import sleep
from typing import List, Optional

# Set up logging
logging.basicConfig(
    filename='logs/nq_tickers_price.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_active_tickers(conn) -> List[str]:
    """Get list of unique tickers from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM nasdaq_100_tickers")
    return [row[0] for row in cursor.fetchall()]

def fetch_price_data(ticker: str, start_date: str, end_date: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch price data with retry mechanism."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval='1d')
            if df.empty:
                logging.warning(f"No data available for {ticker} between {start_date} and {end_date}")
                return None
            return df[['Close', 'Volume']]  # Only keep required columns
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"Failed to fetch data for {ticker} after {retries} attempts: {e}")
                return None
            logging.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            sleep(1)  # Wait before retry

def store_price_data(conn, ticker: str, df: pd.DataFrame):
    """Store price data in the database."""
    try:
        cursor = conn.cursor()
        # Prepare data for insertion
        data = [(ticker, date.strftime('%Y-%m-%d'), row['Close'], row['Volume'])
                for date, row in df.iterrows()]
        # Use INSERT OR REPLACE to handle duplicates
        cursor.executemany("""
            INSERT OR REPLACE INTO nasdaq_100_daily_prices (ticker, date, close, volume)
            VALUES (?, ?, ?, ?)
        """, data)
        conn.commit()
        logging.info(f"Successfully stored {len(data)} price points for {ticker}")
    except Exception as e:
        logging.error(f"Error storing price data for {ticker}: {e}")
        conn.rollback()

def main():
    try:
        # Connect to database
        conn = sqlite3.connect('database/data.db')
        
        # Get unique tickers from your existing table of NASDAQ-100 companies
        tickers = get_active_tickers(conn)
        logging.info(f"Found {len(tickers)} unique tickers to process")
        
        # Define date range for price data
        start_date = '2016-01-01'
        end_date = '2024-12-31'
        
        # Process each active ticker
        for ticker in tickers:
            df = fetch_price_data(ticker, start_date, end_date)
            if df is not None:
                store_price_data(conn, ticker, df)
            sleep(0.5)  # Prevent rate limiting
        
        # --- Now also fetch baseline index data ---
        baseline_tickers = ['SPY', '^NDX']  # SPY ETF and Nasdaq-100 index (Yahoo ticker)
        for ticker in baseline_tickers:
            df = fetch_price_data(ticker, start_date, end_date)
            if df is not None:
                store_price_data(conn, ticker, df)
            sleep(0.5)
        
        conn.close()
        logging.info("Price data fetch completed successfully")
    except Exception as e:
        logging.error(f"Main process failed: {e}")
        raise

if __name__ == "__main__":
    main()
