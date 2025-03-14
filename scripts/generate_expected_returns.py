#!/usr/bin/env python3
"""
Generate Expected Returns

This script calculates expected returns for stocks using the CAPM and Fama-French models
and stores them in the database for use in portfolio optimization.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import logging
import time
from typing import List, Dict, Tuple

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'expected_returns_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_nasdaq100_tickers(db_path: str) -> List[str]:
    """
    Get the list of Nasdaq-100 tickers from the database.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT ticker FROM nasdaq_100_tickers"
    tickers_df = pd.read_sql_query(query, conn)
    conn.close()
    
    return tickers_df['ticker'].tolist()

def get_stock_data_yf(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get historical price data for the given tickers using yfinance.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        DataFrame with stock price data
    """
    all_data = []
    batch_size = 20  # Process in batches to avoid rate limiting
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i+batch_size]
        logging.info(f"Fetching data for tickers {i+1}-{i+len(batch_tickers)} of {len(tickers)}")
        
        # Join tickers with space for yfinance
        ticker_str = " ".join(batch_tickers)
        
        try:
            # Download data for the batch
            data = yf.download(
                ticker_str,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True
            )
            
            # If only one ticker, yfinance returns a different format
            if len(batch_tickers) == 1:
                ticker = batch_tickers[0]
                data_reset = data.reset_index()
                data_reset['ticker'] = ticker
                all_data.append(data_reset)
            else:
                # Process each ticker in the batch
                for ticker in batch_tickers:
                    if ticker in data.columns.levels[0]:
                        ticker_data = data[ticker].reset_index()
                        ticker_data['ticker'] = ticker
                        all_data.append(ticker_data)
            
            # Sleep to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error fetching data for batch {i//batch_size + 1}: {e}")
    
    if not all_data:
        logging.error("No data retrieved from yfinance")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Rename columns to match our expected format
    combined_data = combined_data.rename(columns={
        'Date': 'date',
        'Close': 'close',
        'Volume': 'volume',
        'Open': 'open',
        'High': 'high',
        'Low': 'low'
    })
    
    logging.info(f"Retrieved data for {len(combined_data['ticker'].unique())} stocks from {start_date} to {end_date}")
    return combined_data

def get_market_factors_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get market factors data (Fama-French factors) using yfinance.
    
    For simplicity, we'll use:
    - SPY for market return
    - IWD-IWF for value factor (Value - Growth)
    - IWM-SPY for size factor (Small - Large)
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        DataFrame with market factors data
    """
    # Download data for market factors
    tickers = ['SPY', 'IWD', 'IWF', 'IWM']
    
    try:
        data = yf.download(
            " ".join(tickers),
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True
        )
        
        # Extract close prices
        spy_data = data['SPY']['Close']
        iwd_data = data['IWD']['Close']
        iwf_data = data['IWF']['Close']
        iwm_data = data['IWM']['Close']
        
        # Calculate returns
        spy_returns = spy_data.pct_change().dropna()
        iwd_returns = iwd_data.pct_change().dropna()
        iwf_returns = iwf_data.pct_change().dropna()
        iwm_returns = iwm_data.pct_change().dropna()
        
        # Calculate factors
        market_returns = spy_returns
        value_factor = iwd_returns - iwf_returns  # Value minus Growth
        size_factor = iwm_returns - spy_returns   # Small minus Large
        
        # Combine into a DataFrame
        factors_df = pd.DataFrame({
            'market': market_returns,
            'value': value_factor,
            'size': size_factor
        })
        
        logging.info(f"Retrieved market factors data from {start_date} to {end_date}")
        return factors_df
        
    except Exception as e:
        logging.error(f"Error fetching market factors data: {e}")
        return pd.DataFrame()

def calculate_capm_expected_returns(stock_returns: pd.DataFrame, market_returns: pd.Series, risk_free_rate: float) -> Dict[str, float]:
    """
    Calculate expected returns using the CAPM model.
    
    Args:
        stock_returns: DataFrame with stock returns
        market_returns: Series with market returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Dictionary mapping tickers to expected returns
    """
    expected_returns = {}
    
    # Calculate daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    for ticker in stock_returns.columns:
        # Get stock returns
        stock_ret = stock_returns[ticker].dropna()
        
        # Align market returns with stock returns
        aligned_market = market_returns.loc[stock_ret.index]
        
        # Calculate excess returns
        excess_stock = stock_ret - daily_rf
        excess_market = aligned_market - daily_rf
        
        # Calculate beta using covariance / variance
        if len(excess_stock) > 30:  # Ensure enough data points
            try:
                beta = np.cov(excess_stock, excess_market)[0, 1] / np.var(excess_market)
                
                # Calculate expected return (annualized)
                er = daily_rf * 252 + beta * (market_returns.mean() * 252 - risk_free_rate)
                expected_returns[ticker] = er
            except Exception as e:
                logging.error(f"Error calculating CAPM for {ticker}: {e}")
    
    logging.info(f"Calculated CAPM expected returns for {len(expected_returns)} stocks")
    return expected_returns

def calculate_fama_french_expected_returns(
    stock_returns: pd.DataFrame, 
    factors_df: pd.DataFrame, 
    risk_free_rate: float
) -> Dict[str, float]:
    """
    Calculate expected returns using the Fama-French three-factor model.
    
    Args:
        stock_returns: DataFrame with stock returns
        factors_df: DataFrame with market, size, and value factors
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Dictionary mapping tickers to expected returns
    """
    expected_returns = {}
    
    # Calculate daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate factor means (annualized)
    market_premium = factors_df['market'].mean() * 252
    size_premium = factors_df['size'].mean() * 252
    value_premium = factors_df['value'].mean() * 252
    
    for ticker in stock_returns.columns:
        # Get stock returns
        stock_ret = stock_returns[ticker].dropna()
        
        # Align factors with stock returns
        aligned_factors = factors_df.loc[stock_ret.index]
        
        # Calculate excess returns
        excess_stock = stock_ret - daily_rf
        
        if len(excess_stock) > 60:  # Ensure enough data points
            try:
                # Prepare data for regression
                X = aligned_factors
                y = excess_stock
                
                # Add constant for intercept
                X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
                X.columns = ['const', 'market', 'size', 'value']
                
                # Run regression (manually to avoid statsmodels dependency)
                # X'X
                XX = X.T @ X
                # (X'X)^-1
                XX_inv = np.linalg.inv(XX)
                # (X'X)^-1 X'y
                betas = XX_inv @ (X.T @ y)
                
                # Extract betas
                beta_market = betas[1]
                beta_size = betas[2]
                beta_value = betas[3]
                
                # Calculate expected return (annualized)
                er = (daily_rf * 252 + 
                      beta_market * market_premium + 
                      beta_size * size_premium + 
                      beta_value * value_premium)
                
                expected_returns[ticker] = er
                
            except Exception as e:
                logging.error(f"Error calculating Fama-French for {ticker}: {e}")
    
    logging.info(f"Calculated Fama-French expected returns for {len(expected_returns)} stocks")
    return expected_returns

def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create tables for storing expected returns.
    
    Args:
        conn: SQLite connection
    """
    # Create CAPM table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fundamental_analysis_capm (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        expected_return REAL NOT NULL,
        calculation_date TEXT NOT NULL,
        UNIQUE(ticker)
    )
    """)
    
    # Create Fama-French table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fundamental_analysis_ff (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        expected_return_ff REAL NOT NULL,
        calculation_date TEXT NOT NULL,
        UNIQUE(ticker)
    )
    """)
    
    conn.commit()
    logging.info("Created tables for storing expected returns")

def store_expected_returns(
    conn: sqlite3.Connection, 
    capm_returns: Dict[str, float], 
    ff_returns: Dict[str, float]
) -> None:
    """
    Store expected returns in the database.
    
    Args:
        conn: SQLite connection
        capm_returns: Dictionary with CAPM expected returns
        ff_returns: Dictionary with Fama-French expected returns
    """
    cursor = conn.cursor()
    calculation_date = datetime.now().strftime('%Y-%m-%d')
    
    # Store CAPM expected returns
    capm_data = [(ticker, er, calculation_date) for ticker, er in capm_returns.items()]
    cursor.executemany("""
    INSERT OR REPLACE INTO fundamental_analysis_capm (ticker, expected_return, calculation_date)
    VALUES (?, ?, ?)
    """, capm_data)
    
    # Store Fama-French expected returns
    ff_data = [(ticker, er, calculation_date) for ticker, er in ff_returns.items()]
    cursor.executemany("""
    INSERT OR REPLACE INTO fundamental_analysis_ff (ticker, expected_return_ff, calculation_date)
    VALUES (?, ?, ?)
    """, ff_data)
    
    conn.commit()
    logging.info(f"Stored {len(capm_data)} CAPM expected returns and {len(ff_data)} Fama-French expected returns")

def main():
    """Main function to generate expected returns."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    # Create tables if they don't exist
    create_tables(conn)
    
    # Set date range for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2020-01-01'  # Use 5 years of data for factor analysis
    
    # Get Nasdaq-100 tickers
    tickers = get_nasdaq100_tickers(db_path)
    logging.info(f"Retrieved {len(tickers)} Nasdaq-100 tickers from database")
    
    # Get stock price data
    stock_data = get_stock_data_yf(tickers, start_date, end_date)
    
    if stock_data.empty:
        logging.error("Failed to retrieve stock data. Exiting.")
        return
    
    # Convert to returns dataframe
    pivot_df = stock_data.pivot(index='date', columns='ticker', values='close')
    logging.info(f"Pivot dataframe shape: {pivot_df.shape}")
    
    # Fill missing values with previous values before calculating returns
    pivot_df = pivot_df.fillna(method='ffill')
    
    # Calculate returns and drop first row (which will be NaN)
    stock_returns = pivot_df.pct_change().iloc[1:]
    logging.info(f"Stock returns shape after calculation: {stock_returns.shape}")
    
    # Log first few rows of stock returns for debugging
    logging.info(f"Stock returns first few rows:\n{stock_returns.head()}")
    
    # Get market factors data
    factors_df = get_market_factors_data(start_date, end_date)
    
    if factors_df.empty:
        logging.error("Failed to retrieve market factors data. Exiting.")
        return
    
    # Align stock returns with factor returns
    common_dates = stock_returns.index.intersection(factors_df.index)
    aligned_stock_returns = stock_returns.loc[common_dates]
    aligned_factors = factors_df.loc[common_dates]
    
    # Set risk-free rate (annual)
    risk_free_rate = 0.02  # 2% annual risk-free rate
    
    # Calculate expected returns using CAPM
    capm_returns = calculate_capm_expected_returns(
        aligned_stock_returns, 
        aligned_factors['market'], 
        risk_free_rate
    )
    
    # Calculate expected returns using Fama-French
    ff_returns = calculate_fama_french_expected_returns(
        aligned_stock_returns, 
        aligned_factors, 
        risk_free_rate
    )
    
    # Store expected returns in the database
    if capm_returns or ff_returns:
        store_expected_returns(conn, capm_returns, ff_returns)
    else:
        logging.warning("No expected returns were calculated. Nothing to store in the database.")
    
    # Print summary
    print("\nExpected Returns Summary:")
    print("==================================================")
    print(f"CAPM Model: {len(capm_returns)} stocks")
    print(f"Fama-French Model: {len(ff_returns)} stocks")
    
    # Print top 10 stocks by expected return
    print("\nTop 10 Stocks by Expected Return (CAPM):")
    sorted_capm = sorted(capm_returns.items(), key=lambda x: x[1], reverse=True)
    for ticker, er in sorted_capm[:10]:
        print(f"{ticker}: {er:.4f} ({er*100:.2f}%)")
    
    print("\nTop 10 Stocks by Expected Return (Fama-French):")
    sorted_ff = sorted(ff_returns.items(), key=lambda x: x[1], reverse=True)
    for ticker, er in sorted_ff[:10]:
        print(f"{ticker}: {er:.4f} ({er*100:.2f}%)")
    
    # Close database connection
    conn.close()
    
    logging.info("Expected returns generation completed successfully")
    print("\nExpected returns generation completed successfully")

if __name__ == "__main__":
    main()
