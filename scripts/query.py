import sqlite3
import pandas as pd

def filter_daily_prices_by_constituents(db_path='database/data.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT p.*
    FROM nasdaq_100_daily_prices p
    JOIN nasdaq_100_tickers t 
      ON p.ticker = t.ticker
      AND t.date = (
          SELECT MAX(t2.date)
          FROM nasdaq_100_tickers t2
          WHERE t2.ticker = p.ticker
            AND t2.date <= p.date
      );
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Run the filtering function and inspect the first few rows
filtered_prices = filter_daily_prices_by_constituents()
print(filtered_prices.head())

def count_constituents_per_date(df):
    counts = df.groupby('date')['ticker'].nunique().reset_index(name='num_stocks')
    print(counts.head(10))
    return counts

count_constituents_per_date(filtered_prices)
