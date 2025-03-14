#!/usr/bin/env python3
"""
Integrated Portfolio Strategy

This script combines technical signals with Monte Carlo portfolio optimization
to create an enhanced investment strategy that leverages both technical and
fundamental analysis.

The approach:
1. Calculate technical indicators for all stocks
2. Filter stocks based on technical signals
3. Run Monte Carlo simulation on the filtered universe
4. Adjust portfolio weights based on both technical and fundamental factors
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import json
import yfinance as yf
import time

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'integrated_strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Import functions from technical_signals.py
from technical_signals import (
    compute_all_indicators,
    trading_strategy
)

# Import functions from monte_carlo.py
from monte_carlo import (
    calculate_daily_returns,
    get_expected_returns,
    run_monte_carlo_simulation,
    plot_efficient_frontier,
    export_optimal_portfolios
)

# Import functions from generate_expected_returns.py
from generate_expected_returns import get_nasdaq100_tickers

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

def get_technical_signals(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve stock price data using yfinance, calculate technical indicators, and generate signals.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        DataFrame with technical signals for each stock
    """
    # Get stock price data using yfinance
    price_df = get_stock_data_yf(tickers, start_date, end_date)
    
    if price_df.empty:
        logging.error("Failed to retrieve price data from yfinance")
        return pd.DataFrame()
    
    # Process each ticker to calculate technical indicators and signals
    signals_data = []
    
    for ticker, group in price_df.groupby('ticker'):
        # Sort by date and calculate indicators
        stock_df = group.sort_values(by='date').reset_index(drop=True)
        stock_df = compute_all_indicators(stock_df)
        
        # Drop rows with insufficient indicator data
        stock_df = stock_df.dropna(subset=[
            'SMA_Ratio', 'SMA_Volume_Ratio', 'ATR_Ratio', 
            '20Day_%K', '20Day_%D', 'RSI_20', 
            'MACD_Value', 'MACD_Signal', 'upperband', 'lowerband'
        ])
        
        if stock_df.empty:
            continue
            
        # Generate trading signals
        buy_price, sell_price, trading_signal = trading_strategy(
            prices=stock_df['close'].values,
            SMA_Ratio=stock_df['SMA_Ratio'].values,
            SMA_Volume_Ratio=stock_df['SMA_Volume_Ratio'].values,
            ATR_Ratio=stock_df['ATR_Ratio'].values,
            Day20_K=stock_df['20Day_%K'].values,
            Day20_D=stock_df['20Day_%D'].values,
            RSI_20=stock_df['RSI_20'].values,
            MACD_Value=stock_df['MACD_Value'].values,
            MACD_Signal=stock_df['MACD_Signal'].values,
            upperband=stock_df['upperband'].values,
            lowerband=stock_df['lowerband'].values
        )
        
        # Add signals to the dataframe
        stock_df['buy_signal'] = [1 if not np.isnan(x) else 0 for x in buy_price]
        stock_df['sell_signal'] = [1 if not np.isnan(x) else 0 for x in sell_price]
        stock_df['trading_signal'] = trading_signal
        
        # Get the most recent signal (last 30 days)
        recent_df = stock_df.tail(30).copy()
        
        # Calculate a technical score based on recent signals
        buy_count = recent_df['buy_signal'].sum()
        sell_count = recent_df['sell_signal'].sum()
        
        # Technical score: +1 for each buy signal, -1 for each sell signal
        technical_score = buy_count - sell_count
        
        # Current position (from most recent trading signal)
        current_position = recent_df['trading_signal'].iloc[-1] if len(recent_df) > 0 else 0
        
        signals_data.append({
            'ticker': ticker,
            'technical_score': technical_score,
            'current_position': current_position,
            'last_close': stock_df['close'].iloc[-1] if len(stock_df) > 0 else None,
            'rsi': stock_df['RSI_20'].iloc[-1] if len(stock_df) > 0 else None,
            'macd': stock_df['MACD_Value'].iloc[-1] if len(stock_df) > 0 else None
        })
    
    signals_df = pd.DataFrame(signals_data)
    logging.info(f"Generated technical signals for {len(signals_df)} stocks")
    
    return signals_df

def get_price_data_for_monte_carlo(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get price data for Monte Carlo simulation using yfinance.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        DataFrame with price data in the format expected by Monte Carlo simulation
    """
    # Get stock data using yfinance
    stock_data = get_stock_data_yf(tickers, start_date, end_date)
    
    if stock_data.empty:
        logging.error("Failed to retrieve price data for Monte Carlo simulation")
        return pd.DataFrame()
    
    # Pivot the data to get a DataFrame with dates as index and tickers as columns
    pivot_df = stock_data.pivot(index='date', columns='ticker', values='close')
    
    logging.info(f"Prepared price data for Monte Carlo simulation with {len(pivot_df.columns)} stocks")
    
    return pivot_df

def adjust_expected_returns(
    expected_returns: pd.Series, 
    technical_signals: pd.DataFrame,
    adjustment_weight: float = 0.2
) -> pd.Series:
    """
    Adjust expected returns based on technical signals.
    
    Args:
        expected_returns: Series of expected returns from fundamental analysis
        technical_signals: DataFrame with technical signals
        adjustment_weight: Weight to apply to technical adjustment (0-1)
        
    Returns:
        Adjusted expected returns
    """
    # Create a copy to avoid modifying the original
    adjusted_returns = expected_returns.copy()
    
    # Normalize technical scores to a range of -0.1 to 0.1
    max_score = technical_signals['technical_score'].abs().max()
    if max_score > 0:
        normalized_scores = technical_signals['technical_score'] / max_score * 0.1
    else:
        normalized_scores = technical_signals['technical_score'] * 0
    
    # Create a mapping from ticker to normalized score
    score_map = dict(zip(technical_signals['ticker'], normalized_scores))
    
    # Apply adjustment to expected returns
    for ticker in adjusted_returns.index:
        if ticker in score_map:
            # Apply the technical adjustment with the specified weight
            technical_adjustment = score_map[ticker] * adjustment_weight
            adjusted_returns[ticker] = adjusted_returns[ticker] + technical_adjustment
    
    logging.info(f"Adjusted expected returns for {len(score_map)} stocks based on technical signals")
    
    return adjusted_returns

def filter_investment_universe(
    tickers: List[str],
    technical_signals: pd.DataFrame,
    min_technical_score: int = -5
) -> List[str]:
    """
    Filter the investment universe based on technical signals.
    
    Args:
        tickers: List of all available tickers
        technical_signals: DataFrame with technical signals
        min_technical_score: Minimum technical score to include a stock
        
    Returns:
        Filtered list of tickers
    """
    # Create a set of tickers that meet the technical criteria
    qualified_tickers = set(
        technical_signals[technical_signals['technical_score'] >= min_technical_score]['ticker']
    )
    
    # Filter the original ticker list
    filtered_tickers = [ticker for ticker in tickers if ticker in qualified_tickers]
    
    logging.info(f"Filtered investment universe from {len(tickers)} to {len(filtered_tickers)} stocks")
    
    return filtered_tickers

def store_integrated_portfolio(
    conn: sqlite3.Connection,
    optimization_date: str,
    portfolio_weights: Dict[str, float],
    expected_return: float,
    volatility: float,
    sharpe_ratio: float
) -> None:
    """
    Store the integrated portfolio in the database.
    
    Args:
        conn: Database connection
        optimization_date: Date of optimization
        portfolio_weights: Dictionary mapping tickers to weights
        expected_return: Expected portfolio return
        volatility: Portfolio volatility
        sharpe_ratio: Portfolio Sharpe ratio
    """
    try:
        # Convert weights dictionary to JSON string
        weights_json = json.dumps(portfolio_weights)
        
        # Insert into portfolio_optimization table
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO portfolio_optimization 
        (optimization_date, portfolio_weights, expected_return, volatility, sharpe_ratio)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(
            insert_query, 
            (optimization_date, weights_json, expected_return, volatility, sharpe_ratio)
        )
        conn.commit()
        logging.info("Stored integrated portfolio in database")
    except Exception as e:
        logging.error(f"Error storing portfolio: {e}")

def get_tickers_from_database(db_path: str) -> List[str]:
    """
    Retrieve Nasdaq-100 tickers from the database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        List of Nasdaq-100 tickers
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT DISTINCT ticker FROM nasdaq_100_tickers"
    cursor.execute(query)
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

def main():
    """Main function to run the integrated strategy."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    # Set date range for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2020-01-01'  # Use a fixed start date for consistent analysis
    
    # Step 1: Get Nasdaq-100 tickers from database
    tickers = get_tickers_from_database(db_path)
    logging.info(f"Retrieved {len(tickers)} Nasdaq-100 tickers from database")
    
    # Step 2: Get technical signals using yfinance data
    technical_signals = get_technical_signals(tickers, start_date, end_date)
    
    if technical_signals.empty:
        logging.error("Failed to generate technical signals. Exiting.")
        return
    
    # Step 3: Filter stocks based on technical signals
    filtered_tickers = filter_investment_universe(tickers, technical_signals)
    
    if not filtered_tickers:
        logging.error("No stocks passed the technical filter. Exiting.")
        return
    
    # Step 4: Get price data for Monte Carlo simulation using yfinance
    prices_df = get_price_data_for_monte_carlo(filtered_tickers, start_date, end_date)
    
    if prices_df.empty:
        logging.error("Failed to retrieve price data for Monte Carlo simulation. Exiting.")
        return
    
    # Step 5: Calculate daily returns
    returns_df = calculate_daily_returns(prices_df)
    
    # Step 6: Get expected returns from fundamental analysis
    model = 'fama_french'  # 'capm' or 'fama_french'
    expected_returns = get_expected_returns(conn, model)
    
    # Step 7: Filter expected returns to the filtered universe
    common_tickers = list(set(returns_df.columns) & set(expected_returns.index))
    logging.info(f"Using {len(common_tickers)} stocks with both price data and expected returns")
    
    if not common_tickers:
        logging.error("No stocks have both price data and expected returns. Exiting.")
        return
    
    returns_df = returns_df[common_tickers]
    filtered_expected_returns = expected_returns[common_tickers]
    
    # Step 8: Adjust expected returns based on technical signals
    adjusted_expected_returns = adjust_expected_returns(
        filtered_expected_returns, 
        technical_signals[technical_signals['ticker'].isin(common_tickers)]
    )
    
    # Step 9: Calculate covariance matrix (annualized)
    cov_matrix = returns_df.cov() * 252
    
    # Step 10: Run Monte Carlo simulation with adjusted expected returns
    num_portfolios = 10000
    risk_free_rate = 0.02  # 2% annual risk-free rate
    
    results_df, optimal_portfolios = run_monte_carlo_simulation(
        adjusted_expected_returns,
        cov_matrix,
        num_portfolios,
        risk_free_rate
    )
    
    # Step 11: Plot efficient frontier
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    plot_path = os.path.join(
        results_dir, 
        f'integrated_efficient_frontier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plot_efficient_frontier(results_df, optimal_portfolios, plot_path)
    
    # Step 12: Export optimal portfolios
    export_path = os.path.join(
        results_dir, 
        f'integrated_optimal_portfolios_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    export_optimal_portfolios(
        optimal_portfolios,
        export_path
    )
    
    # Step 13: Store optimal portfolios in database
    optimization_date = datetime.now().strftime('%Y-%m-%d')
    
    # Store max Sharpe ratio portfolio
    max_sharpe_weights = dict(zip(
        adjusted_expected_returns.index,
        optimal_portfolios['max_sharpe'].iloc[3:]
    ))
    store_integrated_portfolio(
        conn,
        optimization_date,
        max_sharpe_weights,
        optimal_portfolios['max_sharpe']['return'],
        optimal_portfolios['max_sharpe']['volatility'],
        optimal_portfolios['max_sharpe']['sharpe_ratio']
    )
    
    # Store min volatility portfolio
    min_vol_weights = dict(zip(
        adjusted_expected_returns.index,
        optimal_portfolios['min_volatility'].iloc[3:]
    ))
    store_integrated_portfolio(
        conn,
        optimization_date,
        min_vol_weights,
        optimal_portfolios['min_volatility']['return'],
        optimal_portfolios['min_volatility']['volatility'],
        optimal_portfolios['min_volatility']['sharpe_ratio']
    )
    
    # Print results
    print("\nIntegrated Portfolio Optimization Results:")
    print("==================================================")
    
    print("\nMax Sharpe Portfolio:")
    print(f"Expected Return: {optimal_portfolios['max_sharpe']['return']:.4f}")
    print(f"Volatility: {optimal_portfolios['max_sharpe']['volatility']:.4f}")
    print(f"Sharpe Ratio: {optimal_portfolios['max_sharpe']['sharpe_ratio']:.4f}")
    
    print("\nTop 10 Holdings:")
    max_sharpe_weights = dict(zip(
        adjusted_expected_returns.index,
        optimal_portfolios['max_sharpe'].iloc[3:]
    ))
    sorted_weights = sorted(max_sharpe_weights.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights[:10]:
        print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    print("\nMin Volatility Portfolio:")
    print(f"Expected Return: {optimal_portfolios['min_volatility']['return']:.4f}")
    print(f"Volatility: {optimal_portfolios['min_volatility']['volatility']:.4f}")
    print(f"Sharpe Ratio: {optimal_portfolios['min_volatility']['sharpe_ratio']:.4f}")
    
    print("\nTop 10 Holdings:")
    min_vol_weights = dict(zip(
        adjusted_expected_returns.index,
        optimal_portfolios['min_volatility'].iloc[3:]
    ))
    sorted_weights = sorted(min_vol_weights.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights[:10]:
        print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Close database connection
    conn.close()
    
    logging.info("Integrated strategy completed successfully")
    print("\nIntegrated strategy completed successfully")

if __name__ == "__main__":
    main()
