import sqlite3
import datetime
import logging
import pandas as pd
import yfinance as yf  # Import yfinance

# Set up logging
logging.basicConfig(
    filename='logs/nq_tickers_yfinance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# List of tickers to process
tickers = [
    'SRCL', 'QRTEA', 'LRCX', 'VRTX', 'DISH', 'COST', 'FB', 'BIDU', 'BBBY', 'DLTR', 'HSIC',
    'INTU', 'TSCO', 'GOOGL', 'AMGN', 'EA', 'LMCA', 'NCLH', 'BKNG', 'MU', 'CERN', 'ADSK',
    'FOXA', 'CSX', 'CA', 'WBA', 'XLNX', 'QCOM', 'YHOO', 'NVDA', 'VIAB', 'NXPI', 'ALXN',
    'PCAR', 'SBAC', 'EBAY', 'AKAM', 'INTC', 'SWKS', 'AAL', 'ADI', 'JD', 'MXIM', 'MYL',
    'DISCK', 'CTSH', 'CMCSA', 'CTXS', 'NTAP', 'TXN', 'AMZN', 'LBTYK', 'AAPL', 'REGN',
    'LLTC', 'CELG', 'DISCA', 'KHC', 'SBUX', 'LMCK', 'NLOK', 'ROST', 'CHTR', 'EXPE', 'MAT',
    'ENDP', 'AMAT', 'INCY', 'FAST', 'ORLY', 'TMUS', 'ATVI', 'ADBE', 'WFM', 'MAR', 'TSLA',
    'PYPL', 'BMRN', 'CHKP', 'NTES', 'PAYX', 'TRIP', 'SIRI', 'ISRG', 'MDLZ', 'TCOM', 'STX',
    'BIIB', 'AVGO', 'VRSK', 'ADP', 'GOOG', 'MSFT', 'ESRX', 'ILMN', 'VOD', 'FOX', 'LBTYA',
    'GILD', 'ULTA', 'NFLX', 'MNST', 'FISV', 'WDC', 'CSCO', 'XRAY', 'MCHP', 'KLAC', 'CTAS',
    'SHPG', 'HAS', 'HOLX', 'IDXX', 'JBHT', 'MELI', 'WYNN', 'SNPS', 'CDNS', 'ALGN', 'WDAY',
    'TTWO', 'ASML', 'PEP', 'XEL', 'UAL', 'VRSN', 'AMD', 'WLTW', 'LULU', 'CSGP', 'SGEN',
    'CPRT', 'ANSS', 'SPLK', 'EXC', 'CDW', 'DOCU', 'ZM', 'DXCM', 'PDD', 'MRNA', 'MRVL',
    'TEAM', 'KDP', 'OKTA', 'AEP', 'PTON', 'MTCH', 'CRWD', 'HON', 'FTNT', 'ABNB', 'PANW',
    'LCID', 'ZS', 'DDOG', 'AZN', 'ODFL', 'CEG', 'META', 'GFS', 'ENPH', 'FANG', 'RIVN',
    'BKR', 'WBD', 'ON', 'GEHC', 'TTD', 'DASH', 'CCEP', 'ROP', 'MDB', 'LIN', 'ARM', 'SMCI',
    'APP', 'AXON', 'PLTR', 'MSTR'
]

# Connect to the SQLite database (ensure you're checking the same file)
conn = sqlite3.connect('database/data.db')
cursor = conn.cursor()

# Create the table for daily prices (storing only close and volume)
cursor.execute('''
CREATE TABLE IF NOT EXISTS nasdaq_100_daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    date TEXT,
    close REAL,
    volume INTEGER
)
''')
conn.commit()
logging.info("Created table 'nasdaq_100_daily_prices' if it didn't exist.")

# Define the date range for historical data
start_date = "2016-01-01"
end_date = "2024-12-31"  # inclusive end date

# Process each ticker
for ticker in tickers:
    try:
        logging.info(f"Fetching data for ticker: {ticker}")
        # Download historical data using yfinance
        hist = yf.download(ticker, start=start_date, end=end_date)
        
        if hist.empty:
            logging.warning(f"No historical data returned for {ticker}.")
            continue
        
        # If columns are multi-indexed, flatten them:
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        
        # Reset index so that 'Date' becomes a column
        hist.reset_index(inplace=True)
        # Rename columns to match our schema
        hist.rename(columns={'Date': 'date', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        rows_inserted = 0
        # Insert each row into the database
        for _, row in hist.iterrows():
            # Format the date as YYYY-MM-DD
            if isinstance(row['date'], pd.Timestamp):
                date_str = row['date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row['date'])
            
            close_price = row.get('close', None)
            vol_val = row.get('volume', 0)
            # If vol_val is a Series (rare), extract its scalar value
            if isinstance(vol_val, pd.Series):
                vol_val = vol_val.iloc[0]
            try:
                volume = int(vol_val)
            except Exception as conv_ex:
                logging.warning(f"Conversion error for volume on {ticker} at {date_str}: {conv_ex}")
                volume = 0
            
            cursor.execute('''
                INSERT INTO nasdaq_100_daily_prices (ticker, date, close, volume)
                VALUES (?, ?, ?, ?)
            ''', (ticker, date_str, close_price, volume))
            rows_inserted += 1
        
        conn.commit()
        logging.info(f"Inserted {rows_inserted} rows for {ticker} into the database.")
    
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}")

# After processing all tickers, count the rows in the table
cursor.execute("SELECT COUNT(*) FROM nasdaq_100_daily_prices")
row_count = cursor.fetchone()[0]
logging.info(f"Total rows in nasdaq_100_daily_prices: {row_count}")
print(f"Total rows inserted: {row_count}")

conn.close()
logging.info("Finished fetching and storing daily close and volume data for all tickers.")



