# backtest_technicals.py

'''
This script contains functions to backtest a technical trading strategy using vectorbt.
The backtest_trading_strategy function loads price data from the database, applies the technical strategy,
and builds a portfolio using vectorbt's Portfolio.from_signals. It then extracts performance metrics.
The update_backtesting_results function inserts the backtesting results into the SQL table backtesting_results.
The optimize_sma_windows function optimizes the SMA window parameters for a given DataFrame of price data.
The deep_dive_analysis function prints detailed portfolio statistics and displays an interactive Plotly performance plot.
'''

import os
import sqlite3
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime
from math import floor
import vectorbt as vbt
import plotly.io as pio

# Set Plotly default renderer to 'browser' for deep dive analysis
pio.renderers.default = 'browser'

# Opt in to future behavior for downcasting
pd.set_option('future.no_silent_downcasting', True)

# Import technical indicator functions from technical_signals.py
from technical_signals import apply_trading_strategy, update_technical_signals

# Set up logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


##############################################
# BACKTESTING & ENHANCED ANALYSIS FUNCTIONS
##############################################

def backtest_trading_strategy(db_path, starting_date='2000-01-01', investment_value=100000):
    """
    For each ticker, load price data from SQL, apply the technical strategy (using original calculations),
    and build a portfolio using vectorbt's Portfolio.from_signals.
    Then extract performance metrics.
    """
    conn = sqlite3.connect(db_path)
    price_df = pd.read_sql_query(
        "SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices", 
        conn, parse_dates=['date']
    )
    conn.close()
    
    results = []
    tickers = price_df['ticker'].unique()
    
    for ticker in tickers:
        df = price_df[(price_df['ticker'] == ticker) & (price_df['date'] >= starting_date)].copy()
        if df.empty:
            continue
        df = df.sort_values('date').reset_index(drop=True)
        df = apply_trading_strategy(df)
        df = df.dropna(subset=['close', 'overall_signal'])
        if df.empty:
            continue
        
        # Define entries and exits: entry when overall_signal turns 1, exit when it turns -1.
        entries = (df['overall_signal'] == 1) & (df['overall_signal'].shift(1) != 1)
        exits = (df['overall_signal'] == -1) & (df['overall_signal'].shift(1) != -1)
        
        # Build the portfolio using vectorbt
        portfolio = vbt.Portfolio.from_signals(
            df['close'],
            entries,
            exits,
            init_cash=investment_value,
            freq='1D'
        )
        
        # Retrieve performance metrics
        total_return = portfolio.total_return()    # e.g., 1.1 means 110%
        profit = total_return * investment_value
        profit_pct = total_return * 100
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        
        # Benchmark: SPY buy-and-hold strategy
        spy_df = price_df[(price_df['ticker'] == 'SPY') & (price_df['date'] >= starting_date)].copy()
        spy_df = spy_df.sort_values('date').reset_index(drop=True)
        if not spy_df.empty:
            spy_return = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[0]) - 1
            spy_profit_pct = spy_return * 100
        else:
            spy_profit_pct = None
        
        benchmark_comparison = profit_pct - spy_profit_pct if spy_profit_pct is not None else None
        
        results.append({
            'Ticker': ticker,
            'Profit Gained': profit,
            'Profit Percentage': profit_pct,
            'Benchmark Profit Comparison': benchmark_comparison,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })
    
    return pd.DataFrame(results)


def update_backtesting_results(db_path, results_df, test_date, strategy_name):
    """
    Inserts backtesting results into the SQL table backtesting_results.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO backtesting_results (test_date, strategy_name, total_return, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?)
        """
        for idx, row in results_df.iterrows():
            sname = f"{strategy_name} - {row['Ticker']}"
            cursor.execute(insert_query, (test_date, sname, row['Profit Gained'], row['Sharpe Ratio'], row['Max Drawdown']))
        conn.commit()
        conn.close()
        print("Backtesting results updated in SQL table backtesting_results.")
    except Exception as e:
        logging.error(f"Error updating backtesting results: {e}")


def optimize_sma_windows(df, sma_short_range, sma_long_range, investment_value=100000):
    """
    Example optimization: loop over SMA window combinations (using original indicator calculations).
    """
    results = []
    for sma_short in sma_short_range:
        for sma_long in sma_long_range:
            if sma_short >= sma_long:
                continue
            
            df_opt = df.copy()
            # Compute SMAs using original methods
            df_opt['SMA_short'] = df_opt['close'].rolling(window=sma_short).mean()
            df_opt['SMA_long'] = df_opt['close'].rolling(window=sma_long).mean()
            signal = df_opt['SMA_short'] > df_opt['SMA_long']
            entries = signal & (~signal.shift(1).fillna(False))
            exits = (~signal) & (signal.shift(1).fillna(False))
            
            portfolio = vbt.Portfolio.from_signals(
                df_opt['close'],
                entries,
                exits,
                init_cash=investment_value,
                freq='1D'
            )
            total_return = portfolio.total_return()
            results.append({
                'sma_short': sma_short,
                'sma_long': sma_long,
                'total_return': total_return,
                'sharpe': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown()
            })
    return pd.DataFrame(results)


def deep_dive_analysis(portfolio):
    """
    Prints detailed portfolio statistics and displays an interactive Plotly performance plot.
    """
    stats = portfolio.stats()
    print("\nDeep Dive Portfolio Statistics:")
    print(stats)
    fig = portfolio.plot()
    fig.update_layout(title="Deep Dive Analysis - Portfolio Performance")
    fig.show()


##############################################
# MAIN EXECUTION
##############################################

if __name__ == "__main__":
    db_path = os.path.join("database", "data.db")
    
    # Step 1: Update technical signals using original indicator calculations
    update_technical_signals(db_path)
    
    # Step 2: Run backtesting via vectorbt
    returns_df = backtest_trading_strategy(db_path, starting_date='2000-01-01', investment_value=100000)
    print("Backtesting Results:")
    print(returns_df)
    
    # Step 3: Update backtesting results in SQL
    test_date = datetime.now().strftime('%Y-%m-%d')
    strategy_name = "Vectorbt Technical Strategy"
    update_backtesting_results(db_path, returns_df, test_date, strategy_name)
    
    # Step 4: Merge Sector Information using yfinance
    tickers = returns_df['Ticker'].unique()
    sector_list = []
    for t in tickers:
        try:
            tickerdata = yf.Ticker(t)
            sector = tickerdata.info.get('sector', 'Unknown')
        except Exception as e:
            sector = 'Unknown'
        sector_list.append(sector)
    sector_df = pd.DataFrame({'Ticker': tickers, 'Sector': sector_list})
    returns_df = returns_df.merge(sector_df, on='Ticker', how='left')
    print("\nResults with Sector Information:")
    print(returns_df)
    
    # Step 5: Display top 25 stocks and sector counts
    ranked_df = returns_df.sort_values(by='Profit Gained', ascending=False).head(25)
    print("\nTop 25 Stocks:")
    print(ranked_df)
    print("\nTop 25 Performing Sectors:")
    print(ranked_df['Sector'].value_counts())
    print("\nTotal Initial Sector Counts:")
    print(returns_df['Sector'].value_counts())
    
    # --- Additional: Parameter Optimization and Deep Dive Analysis ---
    print("\nStarting SMA Parameter Optimization on AAPL:")
    conn = sqlite3.connect(db_path)
    aapl_df = pd.read_sql_query(
        "SELECT date, close, volume FROM nasdaq_100_daily_prices WHERE ticker = 'AAPL' AND date >= '2000-01-01' ORDER BY date", 
        conn, parse_dates=['date']
    )
    conn.close()
    aapl_df = aapl_df.sort_values('date').reset_index(drop=True)
    
    # Define SMA parameter ranges
    sma_short_range = np.arange(3, 15)
    sma_long_range = np.arange(15, 50, 5)
    opt_results = optimize_sma_windows(aapl_df, sma_short_range, sma_long_range, investment_value=100000)
    print("\nOptimization Results for AAPL (SMA windows):")
    print(opt_results.sort_values(by='total_return', ascending=False).head(5))
    
    # Deep dive analysis using best parameters from optimization
    best = opt_results.sort_values(by='total_return', ascending=False).iloc[0]
    best_sma_short = int(best['sma_short'])
    best_sma_long = int(best['sma_long'])
    print(f"\nBest SMA parameters: Short = {best_sma_short}, Long = {best_sma_long}")
    
    # Rebuild signals for AAPL using best SMA parameters (using original indicator calculations)
    aapl_df['SMA_short'] = aapl_df['close'].rolling(window=best_sma_short).mean()
    aapl_df['SMA_long'] = aapl_df['close'].rolling(window=best_sma_long).mean()
    signal = aapl_df['SMA_short'] > aapl_df['SMA_long']
    entries = signal & (~signal.shift(1).fillna(False))
    exits = (~signal) & (signal.shift(1).fillna(False))
    
    best_portfolio = vbt.Portfolio.from_signals(
        aapl_df['close'],
        entries,
        exits,
        init_cash=100000,
        freq='1D'
    )
    deep_dive_analysis(best_portfolio)
