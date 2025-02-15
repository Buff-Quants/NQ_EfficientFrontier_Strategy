## fetch_tickers.py

from nasdaq_100_ticker_history import tickers_as_of 
import pandas as pd
import sqlite3
import logging
import datetime

# Set up logging
logging.basicConfig(filename='logs/nq_tickers.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to the database
conn = sqlite3.connect('database/data.db')
cursor = conn.cursor()

# Create table for NASDAQ-100 tickers
cursor.execute('''
CREATE TABLE IF NOT EXISTS nasdaq_100_tickers (
    id INTEGER PRIMARY KEY,
    date TEXT,
    ticker TEXT
)
''')
conn.commit()

# Fetch and store NASDAQ-100 tickers
exact_dates = pd.date_range(start='2016-01-01', end='2024-12-31', freq='Q')
for date in exact_dates:
    tickers = tickers_as_of(date.year, date.month, date.day)
    for ticker in tickers:
        cursor.execute("INSERT INTO nasdaq_100_tickers (date, ticker) VALUES (?, ?)", (date.strftime('%Y-%m-%d'), ticker))
        logging.info(f"Inserted ticker {ticker} for date {date.strftime('%Y-%m-%d')}")

conn.commit()
conn.close()

logging.info("NASDAQ-100 tickers have been successfully stored in the database.")
