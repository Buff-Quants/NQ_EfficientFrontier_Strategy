## fetch_tickers.py

import pandas as pd
import sqlite3
import logging
from datetime import datetime
from config import DB_PATH, LOG_DIR, LOG_LEVEL, LOG_FILE

# Configure logging (log file for ticker fetching)
log_file = f"{LOG_DIR}/nq_tickers.log"
logging.basicConfig(
    filename=log_file,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_database() -> sqlite3.Connection:
    """
    Create a database connection and ensure the nasdaq_100_tickers table exists.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nasdaq_100_tickers (
                id INTEGER PRIMARY KEY,
                date TEXT,
                ticker TEXT,
                UNIQUE(date, ticker)
            )
        ''')
        conn.commit()
        return conn
    except Exception as e:
        logging.error(f"Database setup failed in fetch_tickers: {e}")
        raise

def get_nasdaq100_tickers() -> list:
    """
    Return a hardcoded list of current NASDAQ-100 tickers (as of March 2025).
    """
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'ADBE', 
        'COST', 'PEP', 'CSCO', 'NFLX', 'CMCSA', 'AMD', 'TMUS', 'INTC', 'INTU', 'QCOM', 
        'TXN', 'AMAT', 'AMGN', 'HON', 'SBUX', 'ISRG', 'ADI', 'MDLZ', 'GILD', 'REGN', 
        'ADP', 'VRTX', 'PANW', 'KLAC', 'LRCX', 'SNPS', 'ASML', 'CDNS', 'MRVL', 'BKNG', 
        'ABNB', 'ADSK', 'ORLY', 'FTNT', 'CTAS', 'MELI', 'MNST', 'PAYX', 'KDP', 'PCAR', 
        'CRWD', 'DXCM', 'CHTR', 'LULU', 'NXPI', 'MCHP', 'WDAY', 'CPRT', 'ODFL', 'ROST', 
        'FAST', 'EXC', 'BIIB', 'CSGP', 'ANSS', 'CTSH', 'DDOG', 'IDXX', 'VRSK', 'TEAM', 
        'DLTR', 'ILMN', 'ZS', 'ALGN', 'MTCH', 'FANG', 'ENPH', 'GEHC', 'DASH', 'SGEN', 
        'SIRI', 'CCEP', 'SPLK', 'TTWO', 'VRSN', 'SWKS', 'AEP', 'WBD', 'XEL', 'CSX', 
        'FISV', 'ATVI', 'MDB', 'PYPL', 'LCID', 'RIVN', 'TTD', 'SMCI', 'PLTR', 'MSTR'
    ]
    return tickers

def fetch_and_store_tickers(conn: sqlite3.Connection) -> None:
    """
    Fetch and store current NASDAQ-100 tickers into the nasdaq_100_tickers table.
    """
    cursor = conn.cursor()
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        tickers = get_nasdaq100_tickers()
        if not tickers:
            logging.error("No tickers retrieved")
            return
        
        for ticker in tickers:
            cursor.execute("""
                INSERT OR REPLACE INTO nasdaq_100_tickers (date, ticker)
                VALUES (?, ?)
            """, (current_date, ticker))
        conn.commit()
        logging.info(f"Successfully stored {len(tickers)} tickers for {current_date}")
    except Exception as e:
        logging.error(f"Error in fetch_and_store_tickers: {e}")
        conn.rollback()
        raise

def main():
    """Main function to set up the database and fetch tickers."""
    try:
        conn = setup_database()
        fetch_and_store_tickers(conn)
        conn.close()
        logging.info("Ticker fetch process completed successfully")
    except Exception as e:
        logging.error(f"Main process in fetch_tickers failed: {e}")
        raise

if __name__ == "__main__":
    main()
