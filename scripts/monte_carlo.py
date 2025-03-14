#!/usr/bin/env python
"""
Monte Carlo simulation for portfolio optimization.

This script performs Monte Carlo simulation to find the optimal portfolio weights
that maximize the Sharpe ratio while considering the risk-return tradeoff.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import time

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'monte_carlo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_stock_data(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve stock price data from the database.
    
    Args:
        conn: SQLite connection
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        
    Returns:
        DataFrame with stock prices
    """
    print(f"Retrieving stock data from {start_date} to {end_date}...")
    query = f"""
    SELECT date, ticker, close
    FROM nasdaq_100_daily_prices
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date, ticker
    """
    
    df = pd.read_sql_query(query, conn)
    print(f"Retrieved {len(df)} price records")
    
    # Check for duplicate entries
    duplicates = df.duplicated(subset=['date', 'ticker'])
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate entries. Removing duplicates...")
        df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')
    
    # Pivot the data to have tickers as columns and dates as index
    pivot_df = df.pivot(index='date', columns='ticker', values='close')
    
    # Forward fill missing values (if any)
    pivot_df = pivot_df.ffill().bfill()
    
    logging.info(f"Retrieved price data for {pivot_df.shape[1]} stocks from {start_date} to {end_date}")
    print(f"Processed price data for {pivot_df.shape[1]} stocks across {pivot_df.shape[0]} days")
    
    return pivot_df

def get_expected_returns(conn, model: str = 'fama_french') -> pd.Series:
    """
    Retrieve expected returns from fundamental analysis.
    
    Args:
        conn: SQLite connection
        model: 'capm' or 'fama_french'
        
    Returns:
        Series with expected returns for each ticker
    """
    if model.lower() == 'capm':
        query = """
        SELECT ticker, expected_return
        FROM fundamental_analysis_capm
        ORDER BY ticker
        """
        df = pd.read_sql_query(query, conn)
        return_col = 'expected_return'
    else:  # Fama-French
        query = """
        SELECT ticker, expected_return_ff
        FROM fundamental_analysis_ff
        ORDER BY ticker
        """
        df = pd.read_sql_query(query, conn)
        return_col = 'expected_return_ff'
    
    # Convert to annualized decimal returns if needed
    if df[return_col].mean() > 1:  # If returns are in percentage
        df[return_col] = df[return_col] / 100
    
    logging.info(f"Retrieved expected returns for {len(df)} stocks using {model} model")
    
    return pd.Series(df[return_col].values, index=df['ticker'])

def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.
    
    Args:
        prices: DataFrame with stock prices
        
    Returns:
        DataFrame with daily returns
    """
    return prices.pct_change().dropna()

def generate_random_weights(n_assets: int) -> np.ndarray:
    """
    Generate random weights that sum to 1.
    
    Args:
        n_assets: Number of assets
        
    Returns:
        Array of random weights
    """
    weights = np.random.random(n_assets)
    return weights / np.sum(weights)

def calculate_portfolio_metrics(
    weights: np.ndarray, 
    expected_returns: np.ndarray, 
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02
) -> Tuple[float, float, float]:
    """
    Calculate portfolio return, volatility, and Sharpe ratio.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Tuple of (return, volatility, Sharpe ratio)
    """
    # Portfolio return
    portfolio_return = np.sum(weights * expected_returns)
    
    # Portfolio volatility (annualized)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def run_monte_carlo_simulation(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_portfolios: int = 10000,
    risk_free_rate: float = 0.02
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run Monte Carlo simulation to generate random portfolios.
    
    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        num_portfolios: Number of portfolios to simulate
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        DataFrame with portfolio metrics and weights, and dictionary with optimal portfolios
    """
    n_assets = len(expected_returns)
    results = np.zeros((num_portfolios, 3 + n_assets))
    
    start_time = time.time()
    logging.info(f"Starting Monte Carlo simulation with {num_portfolios} portfolios")
    
    for i in range(num_portfolios):
        weights = generate_random_weights(n_assets)
        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix, risk_free_rate
        )
        
        results[i, 0] = portfolio_return
        results[i, 1] = portfolio_volatility
        results[i, 2] = sharpe_ratio
        results[i, 3:] = weights
        
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            logging.info(f"Completed {i} portfolios in {elapsed:.2f} seconds")
    
    elapsed = time.time() - start_time
    logging.info(f"Monte Carlo simulation completed in {elapsed:.2f} seconds")
    
    # Convert results to DataFrame
    columns = ['return', 'volatility', 'sharpe_ratio'] + list(expected_returns.index)
    results_df = pd.DataFrame(results, columns=columns)
    
    # Find optimal portfolios
    optimal_portfolios = {
        'max_sharpe': results_df.iloc[results_df['sharpe_ratio'].idxmax()],
        'min_volatility': results_df.iloc[results_df['volatility'].idxmin()],
        'max_return': results_df.iloc[results_df['return'].idxmax()]
    }
    
    return results_df, optimal_portfolios

def plot_efficient_frontier(
    results_df: pd.DataFrame,
    optimal_portfolios: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the efficient frontier and optimal portfolios.
    
    Args:
        results_df: DataFrame with portfolio metrics
        optimal_portfolios: Dictionary with optimal portfolios
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all portfolios
    plt.scatter(
        results_df['volatility'], 
        results_df['return'], 
        c=results_df['sharpe_ratio'], 
        cmap='viridis', 
        alpha=0.3,
        s=10
    )
    
    # Plot optimal portfolios
    plt.scatter(
        optimal_portfolios['max_sharpe']['volatility'],
        optimal_portfolios['max_sharpe']['return'],
        c='red',
        marker='*',
        s=300,
        label='Maximum Sharpe Ratio'
    )
    
    plt.scatter(
        optimal_portfolios['min_volatility']['volatility'],
        optimal_portfolios['min_volatility']['return'],
        c='green',
        marker='o',
        s=200,
        label='Minimum Volatility'
    )
    
    plt.scatter(
        optimal_portfolios['max_return']['volatility'],
        optimal_portfolios['max_return']['return'],
        c='blue',
        marker='^',
        s=200,
        label='Maximum Return'
    )
    
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Annualized)')
    plt.ylabel('Expected Return (Annualized)')
    plt.title('Efficient Frontier - Monte Carlo Simulation')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved efficient frontier plot to {save_path}")
    
    plt.show()

def store_optimal_portfolio(conn, portfolio_data: pd.Series, portfolio_type: str) -> bool:
    """
    Store optimal portfolio weights in the database.
    
    Args:
        conn: SQLite connection
        portfolio_data: Series with portfolio metrics and weights
        portfolio_type: Type of portfolio ('max_sharpe', 'min_volatility', 'max_return')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_optimization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_type TEXT,
            date TEXT,
            expected_return REAL,
            volatility REAL,
            sharpe_ratio REAL,
            weights TEXT
        )
        """)
        
        # Extract portfolio metrics
        expected_return = portfolio_data['return']
        volatility = portfolio_data['volatility']
        sharpe_ratio = portfolio_data['sharpe_ratio']
        
        # Extract weights and convert to JSON
        weights = portfolio_data.iloc[3:].to_dict()
        weights_json = pd.Series(weights).to_json()
        
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if portfolio type already exists for today
        cursor.execute("""
        SELECT id FROM portfolio_optimization
        WHERE portfolio_type = ? AND date = ?
        """, (portfolio_type, current_date))
        
        existing_entry = cursor.fetchone()
        
        if existing_entry:
            # Update existing entry
            cursor.execute("""
            UPDATE portfolio_optimization
            SET expected_return = ?, volatility = ?, sharpe_ratio = ?, weights = ?
            WHERE portfolio_type = ? AND date = ?
            """, (expected_return, volatility, sharpe_ratio, weights_json, portfolio_type, current_date))
        else:
            # Insert new entry
            cursor.execute("""
            INSERT INTO portfolio_optimization (
                portfolio_type, date, expected_return, volatility, sharpe_ratio, weights
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (portfolio_type, current_date, expected_return, volatility, sharpe_ratio, weights_json))
        
        conn.commit()
        logging.info(f"Stored {portfolio_type} portfolio in database")
        return True
    
    except Exception as e:
        logging.error(f"Error storing portfolio: {e}")
        return False

def export_optimal_portfolios(optimal_portfolios: Dict, save_path: str) -> None:
    """
    Export optimal portfolios to CSV.
    
    Args:
        optimal_portfolios: Dictionary with optimal portfolios
        save_path: Path to save the CSV file
    """
    # Create a DataFrame to store the results
    results = []
    
    for portfolio_type, portfolio_data in optimal_portfolios.items():
        # Extract portfolio metrics
        portfolio_return = portfolio_data['return']
        portfolio_volatility = portfolio_data['volatility']
        portfolio_sharpe = portfolio_data['sharpe_ratio']
        
        # Extract weights
        weights = portfolio_data.iloc[3:].to_dict()
        
        # Add to results
        row = {
            'portfolio_type': portfolio_type,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            **{f'weight_{ticker}': weight for ticker, weight in weights.items()}
        }
        
        results.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(save_path, index=False)
    logging.info(f"Exported optimal portfolios to {save_path}")

def main():
    """Main function to run the Monte Carlo simulation."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    # Set date range for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2020-01-01'  # Use a fixed start date for consistent analysis
    
    # Get stock price data
    prices_df = get_stock_data(conn, start_date, end_date)
    
    # Calculate daily returns
    returns_df = calculate_daily_returns(prices_df)
    
    # Get expected returns from fundamental analysis
    model = 'fama_french'  # 'capm' or 'fama_french'
    expected_returns = get_expected_returns(conn, model)
    
    # Filter stocks to include only those with expected returns
    common_tickers = list(set(returns_df.columns) & set(expected_returns.index))
    logging.info(f"Using {len(common_tickers)} stocks with both price data and expected returns")
    
    returns_df = returns_df[common_tickers]
    expected_returns = expected_returns[common_tickers]
    
    # Calculate covariance matrix (annualized)
    cov_matrix = returns_df.cov() * 252
    
    # Run Monte Carlo simulation
    num_portfolios = 10000
    risk_free_rate = 0.02  # 2% annual risk-free rate
    
    results_df, optimal_portfolios = run_monte_carlo_simulation(
        expected_returns,
        cov_matrix,
        num_portfolios,
        risk_free_rate
    )
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot efficient frontier
    plot_path = os.path.join(results_dir, f'efficient_frontier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plot_efficient_frontier(results_df, optimal_portfolios, plot_path)
    
    # Store optimal portfolios in database
    for portfolio_type, portfolio_data in optimal_portfolios.items():
        store_optimal_portfolio(conn, portfolio_data, portfolio_type)
    
    # Export optimal portfolios to CSV
    csv_path = os.path.join(results_dir, f'optimal_portfolios_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    export_optimal_portfolios(optimal_portfolios, csv_path)
    
    # Print summary
    print("\nPortfolio Optimization Results:")
    print("=" * 50)
    
    for portfolio_type, portfolio_data in optimal_portfolios.items():
        print(f"\n{portfolio_type.replace('_', ' ').title()} Portfolio:")
        print(f"Expected Return: {portfolio_data['return']:.4f}")
        print(f"Volatility: {portfolio_data['volatility']:.4f}")
        print(f"Sharpe Ratio: {portfolio_data['sharpe_ratio']:.4f}")
        
        # Print top 10 weights
        weights = portfolio_data.iloc[3:].sort_values(ascending=False)
        print("\nTop 10 Holdings:")
        for ticker, weight in weights.head(10).items():
            print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Close connection
    conn.close()
    
    logging.info("Monte Carlo simulation completed successfully")

if __name__ == "__main__":
    main()