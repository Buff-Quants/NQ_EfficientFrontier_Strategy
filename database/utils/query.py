import sqlite3

# Connect to your database
conn = sqlite3.connect('database/data.db')
cursor = conn.cursor()

# Execute a query to get all unique tickers
cursor.execute("SELECT DISTINCT ticker FROM nasdaq_100_tickers")
tickers_data = cursor.fetchall()

# Convert the results into a list of tickers
tickers = [row[0] for row in tickers_data]

# Optionally, print the tickers to verify
print(tickers)

# Close the connection if you're done with it
conn.close()

import sqlite3

# Connect to your database
conn = sqlite3.connect('database/data.db')
cursor = conn.cursor()

# Drop the table if it exists
cursor.execute("DROP TABLE IF EXISTS nasdaq_100_daily_prices")
conn.commit()

conn.close()
print("Table 'nasdaq_100_daily_prices' has been dropped.")



import sqlite3
import pandas as pd

def inspect_company_rows(db_path='database/data.db', table_name='nasdaq_100_daily_prices_combined', expected_rows=2263):
    """
    Connects to the SQLite database and inspects the specified table,
    grouping by the company (or ticker) column. It outputs the companies
    that have fewer than the expected number of rows.
    
    Parameters:
      - db_path (str): Path to the SQLite database.
      - table_name (str): Name of the table to query.
      - expected_rows (int): Expected number of rows per company.
      
    Returns:
      - pd.DataFrame: A DataFrame with companies having less than expected_rows.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        # Adjust the column name below ("company") if your table uses "ticker" instead.
        query = f"SELECT company, COUNT(*) as row_count FROM {table_name} GROUP BY company"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print("Error reading from the database:", e)
        return None
    
    # Filter for companies with fewer than the expected number of rows
    incomplete = df[df['row_count'] < expected_rows]
    
    if incomplete.empty:
        print(f"All companies have at least {expected_rows} rows.")
    else:
        print("Companies with fewer than {} rows:".format(expected_rows))
        print(incomplete)
    
    return incomplete

# Example usage:
incomplete_companies = inspect_company_rows()
