#!/usr/bin/env python
"""
run_pipeline.py

This script runs the entire data pipeline, including:
- Setting up the database schema,
- Fetching tickers,
- Fetching price data,
- Cleaning up duplicate entries (in fundamental tables),
- Running fundamental analysis, and
- Computing technical signals, as well as backtesting.
All configuration values are centralized in config.py.
"""

import logging
from datetime import datetime
from config import DB_PATH, SCHEMA_PATH, LOG_FILE, LOG_LEVEL
from scripts.db_utils import create_connection, setup_database
from scripts.fetch_tickers import main as fetch_tickers_main
from scripts.fetch_price import main as fetch_price_main
from scripts.cleanup_database import main as cleanup_main
from scripts.fundamentals import main as run_fundamentals
from scripts.technical_signals import update_technical_signals
from scripts.backtest_technicals import backtest_trading_strategy, update_backtesting_results

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
    fetch_tickers_main()  # This writes tickers into nasdaq_100_tickers
    
    # Step 3: Fetch price data for tickers and baseline indices
    logging.info("Fetching price data...")
    fetch_price_main()  # This writes price data into nasdaq_100_daily_prices
    
    # Step 4: Clean up duplicate entries in fundamental analysis tables
    logging.info("Cleaning up duplicate entries...")
    cleanup_main()  # Cleans up duplicates in fundamental_analysis_capm and fundamental_analysis_ff
    
    # Step 5: Run fundamental analysis and store the results
    logging.info("Running fundamental analysis...")
    run_fundamentals()  # Processes all tickers based on price data
    
    # Step 6: Compute technical signals and store them in the database
    logging.info("Computing technical signals...")
    update_technical_signals(DB_PATH)
    
    # Step 7: Backtesting the trading strategy
    logging.info("Running backtesting analysis...")
    returns_df = backtest_trading_strategy()  
    logging.info("Backtesting completed.")
    
    test_date = datetime.now().strftime('%Y-%m-%d')
    strategy_name = "Technical Strategy"
    update_backtesting_results(DB_PATH, returns_df, test_date, strategy_name)
    
    logging.info("=== Data Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
