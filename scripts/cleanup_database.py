#!/usr/bin/env python
"""
Script to clean up duplicate entries in the fundamental analysis tables.
"""

import os
import sqlite3
import logging
from datetime import datetime

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def cleanup_table(conn, table_name):
    """
    Remove duplicate entries from the specified table, keeping only the most recent entry for each ticker.
    """
    cursor = conn.cursor()
    
    # Get list of tickers with duplicates
    cursor.execute(f"""
    SELECT ticker, COUNT(*) as count
    FROM {table_name}
    GROUP BY ticker
    HAVING count > 1
    """)
    
    duplicates = cursor.fetchall()
    logging.info(f"Found {len(duplicates)} tickers with duplicate entries in {table_name}")
    
    # For each ticker with duplicates, keep only the most recent entry
    for ticker, count in duplicates:
        logging.info(f"Cleaning up {count} duplicate entries for {ticker}")
        
        # Create a temporary table with the most recent entry for each ticker
        cursor.execute(f"""
        CREATE TEMPORARY TABLE temp_{table_name} AS
        SELECT *
        FROM {table_name}
        WHERE ticker = ?
        ORDER BY analysis_date DESC
        LIMIT 1
        """, (ticker,))
        
        # Delete all entries for this ticker
        cursor.execute(f"""
        DELETE FROM {table_name}
        WHERE ticker = ?
        """, (ticker,))
        
        # Insert the most recent entry back
        cursor.execute(f"""
        INSERT INTO {table_name}
        SELECT * FROM temp_{table_name}
        """)
        
        # Drop the temporary table
        cursor.execute(f"""
        DROP TABLE temp_{table_name}
        """)
    
    conn.commit()
    logging.info(f"Cleanup completed for {table_name}")

def main():
    """Main function to clean up the database."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    # Clean up CAPM table
    cleanup_table(conn, 'fundamental_analysis_capm')
    
    # Clean up Fama-French table
    cleanup_table(conn, 'fundamental_analysis_ff')
    
    # Close connection
    conn.close()
    
    logging.info("Database cleanup completed successfully")

if __name__ == "__main__":
    main()
