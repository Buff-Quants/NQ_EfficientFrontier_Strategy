"""
run_pipeline.py

This script runs the entire data pipeline, including:
- Setting up the database schema,
- Fetching tickers,
- Fetching price data,
- Cleaning up duplicate entries,
- Running fundamental analysis, and
- Computing technical signals.
"""

import logging
from config import DB_PATH, SCHEMA_PATH, LOG_FILE, LOG_LEVEL
from scripts.db_utils import create_connection, setup_database
from scripts.fetch_tickers import main as fetch_tickers_main
from scripts.fetch_price import main as fetch_price_main
from scripts.cleanup_database import main as cleanup_main
from scripts.fundamentals import main as run_fundamentals
from scripts.technical_signals import update_technical_signals

# Configure logging for the pipeline
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    logging.info("=== Starting Data Pipeline ===")
    
    # Step 1: Set up the database schema
    logging.info("Setting up database schema...")
    conn = create_connection(DB_PATH)
    setup_database(conn, SCHEMA_PATH)
    conn.close()
    
    # Step 2: Fetch tickers and store them in the database
    logging.info("Fetching tickers...")
    fetch_tickers_main()  # This function writes tickers to the appropriate table
    
    # Step 3: Fetch price data for tickers and baseline indices
    logging.info("Fetching price data...")
    fetch_price_main()  # This function writes price data to the database
    
    # Step 4: Run fundamental analysis and store the results
    logging.info("Running fundamental analysis...")
    run_fundamentals()  # Make sure fundamentals.py is set to process all tickers as desired.
    
    # Step 5: Compute technical signals and store them in the database
    logging.info("Computing technical signals...")
    update_technical_signals(DB_PATH)  # Pass DB_PATH as needed.

    # Step 6: Clean up duplicate entries in the database
    logging.info("Running cleanup on database...")
    cleanup_main()  # Cleans up duplicate entries and runs VACUUM
    
    logging.info("=== Data Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
