## fetch_price.py

import sqlite3
import datetime
import logging
import pandas as pd
import yfinance as yf
from time import sleep
from typing import List, Optional
from config import DB_PATH, LOG_DIR, LOG_LEVEL, LOG_FILE

# Configure logging (log file dedicated for price fetching)
log_file = f"{LOG_DIR}/nq_tickers_price.log"
logging.basicConfig(
    filename=log_file,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_active_tickers(conn: sqlite3.Connection) -> List[str]:
    """Get list of unique tickers from the nasdaq_100_tickers table."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM nasdaq_100_tickers")
    return [row[0] for row in cursor.fetchall()]

def fetch_price_data(ticker: str, start_date: str, end_date: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch price data for a given ticker using yfinance with a retry mechanism."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval='1d')
            if df.empty:
                logging.warning(f"No data available for {ticker} between {start_date} and {end_date}")
                return None
            return df[['Close', 'Volume']]
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"Failed to fetch data for {ticker} after {retries} attempts: {e}")
                return None
            logging.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            sleep(1)

def store_price_data(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """Store fetched price data for a ticker into the nasdaq_100_daily_prices table."""
    try:
        cursor = conn.cursor()
        # Prepare data for insertion: (ticker, date, close, volume)
        data = [(ticker, date.strftime('%Y-%m-%d'), row['Close'], row['Volume'])
                for date, row in df.iterrows()]
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
    """Main function to fetch and store price data for active tickers and baseline indices."""
    try:
        conn = sqlite3.connect(DB_PATH)
        tickers = get_active_tickers(conn)
        logging.info(f"Found {len(tickers)} unique tickers to process")
        
        # Define the date range for fetching price data
        start_date = '2016-01-01'
        end_date = '2024-12-31'
        
        # Process each active ticker
        for ticker in tickers:
            df = fetch_price_data(ticker, start_date, end_date)
            if df is not None:
                store_price_data(conn, ticker, df)
            sleep(0.5)  # Pause to help avoid rate limiting
        
        # Process baseline tickers (e.g., SPY, ^NDX)
        baseline_tickers = ['SPY', '^NDX']
        for ticker in baseline_tickers:
            df = fetch_price_data(ticker, start_date, end_date)
            if df is not None:
                store_price_data(conn, ticker, df)
            sleep(0.5)
        
        conn.close()
        logging.info("Price data fetch completed successfully")
    except Exception as e:
        logging.error(f"Main process in fetch_price failed: {e}")
        raise

if __name__ == "__main__":
    main()
