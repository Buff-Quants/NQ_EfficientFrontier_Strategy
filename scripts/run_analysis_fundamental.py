#!/usr/bin/env python3
"""
Script to run fundamental analysis on a set of tickers and store results in the database.
"""

import os
import sys
import sqlite3
import logging
import pandas as pd
from datetime import datetime
from fundamentals import run_fundamental_analysis, store_fundamental_results

# Set up logging for the analysis script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/run_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_analysis_for_tickers(tickers, start_date, end_date, model='capm'):
    """Run fundamental analysis for a list of tickers and store results in the database."""
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, ticker in enumerate(tickers):
        logging.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
        try:
            result = run_fundamental_analysis(conn, ticker, start_date, end_date, model)
            if result:
                store_fundamental_results(conn, result)
                results.append(result)
                success_count += 1
                logging.info(f"Successfully analyzed {ticker}")
            else:
                error_count += 1
                logging.warning(f"No results for {ticker}")
        except Exception as e:
            error_count += 1
            logging.error(f"Error analyzing {ticker}: {e}")
    conn.close()
    logging.info(f"Analysis completed. Success: {success_count}, Errors: {error_count}")
    return results

def main():
    """Main function to run the analysis for all tickers."""
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM nasdaq_100_daily_prices ORDER BY ticker", conn)
    tickers = tickers_df['ticker'].tolist()
    logging.info(f"Found {len(tickers)} tickers to analyze")
    conn.close()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2020-01-01'
    model = 'fama_french'  # or 'capm'
    
    results = run_analysis_for_tickers(tickers, start_date, end_date, model)
    
    # Print summary
    if results:
        print("\nAnalysis Results Summary:")
        print(f"Total tickers analyzed: {len(results)}")
        avg_beta = sum(r['beta'] for r in results if r['beta'] is not None) / len(results)
        avg_alpha = sum(r['ff_factor1'] for r in results if r['ff_factor1'] is not None) / len(results)
        avg_expected_return = sum(r['expected_return'] for r in results if r['expected_return'] is not None) / len(results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in results if r['sharpe_ratio'] is not None) / len(results)
        print(f"Average Beta: {avg_beta:.4f}")
        print(f"Average Alpha (annualized): {avg_alpha:.4f}")
        print(f"Average Expected Return (annualized): {avg_expected_return:.4f}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
    else:
        print("No analysis results to display.")

if __name__ == "__main__":
    main()
