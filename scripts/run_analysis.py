#!/usr/bin/env python3
"""
Script to run fundamental analysis on a set of tickers.
"""

import os
import sys
import sqlite3
import logging
import pandas as pd
from datetime import datetime
from fundamentals import run_fundamental_analysis, store_fundamental_results

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/run_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_analysis_for_tickers(tickers, start_date, end_date, model='capm'):
    """Run fundamental analysis for a list of tickers."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, ticker in enumerate(tickers):
        logging.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
        try:
            # Run fundamental analysis
            result = run_fundamental_analysis(conn, ticker, start_date, end_date, model)
            
            if result:
                # Store results
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
    
    # Close connection
    conn.close()
    
    logging.info(f"Analysis completed. Success: {success_count}, Errors: {error_count}")
    return results

def main():
    """Main function to run the analysis."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    # Get tickers from database
    use_all_tickers = True  # Set to True to process all tickers
    
    if use_all_tickers:
        tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM nasdaq_100_daily_prices ORDER BY ticker", conn)
        tickers = tickers_df['ticker'].tolist()
        logging.info(f"Found {len(tickers)} tickers to analyze")
    else:
        # Use a subset of tickers for testing
        tickers = ['NVDA', 'TSLA', 'AVGO']  # Test with different tickers
    
    conn.close()
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2020-01-01'  # Use a fixed start date for consistent analysis
    
    # Choose model
    model = 'fama_french'  # 'capm' or 'fama_french'
    
    # Run analysis
    results = run_analysis_for_tickers(tickers, start_date, end_date, model)
    
    # Print summary
    if results:
        print("\nAnalysis Results Summary:")
        print(f"Total tickers analyzed: {len(results)}")
        
        # Calculate average metrics
        avg_beta = sum(r['beta'] for r in results) / len(results)
        avg_alpha = sum(r['alpha'] for r in results) / len(results)
        avg_expected_return_capm = sum(r['expected_return_capm'] for r in results) / len(results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in results) / len(results)
        
        print(f"Average Beta: {avg_beta:.4f}")
        print(f"Average Alpha (annualized): {avg_alpha:.4f}")
        print(f"Average Expected Return (CAPM, annualized): {avg_expected_return_capm:.4f}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
        
        if model.lower() == 'fama_french':
            # Calculate Fama-French averages
            avg_smb_beta = sum(r.get('smb_beta', 0) for r in results) / len(results)
            avg_hml_beta = sum(r.get('hml_beta', 0) for r in results) / len(results)
            avg_expected_return_ff = sum(r.get('expected_return_ff', 0) for r in results) / len(results)
            
            print(f"Average SMB Beta: {avg_smb_beta:.4f}")
            print(f"Average HML Beta: {avg_hml_beta:.4f}")
            print(f"Average Expected Return (FF, annualized): {avg_expected_return_ff:.4f}")
        
        # Find top performers
        results_sorted_by_sharpe = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        results_sorted_by_return = sorted(results, key=lambda x: x['expected_return_capm'], reverse=True)
        
        print("\nTop 5 tickers by Sharpe Ratio:")
        for i, r in enumerate(results_sorted_by_sharpe[:5]):
            print(f"{i+1}. {r['ticker']}: {r['sharpe_ratio']:.4f}")
        
        print("\nTop 5 tickers by Expected Return (CAPM):")
        for i, r in enumerate(results_sorted_by_return[:5]):
            print(f"{i+1}. {r['ticker']}: {r['expected_return_capm']:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'fundamental_analysis_{model}_{timestamp}.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("No analysis results to display.")

if __name__ == "__main__":
    main()
