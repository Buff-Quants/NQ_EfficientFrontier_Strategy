import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from math import floor, sqrt
import yfinance as yf  # For fetching sector info
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Set up logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#########################
# TECHNICAL INDICATOR FUNCTIONS (using 'close' column)
#########################

def compute_sma_signals(df, short_window=5, long_window=20):
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    df['SMA_Ratio'] = df['SMA_long'] / df['SMA_short']
    df['SMA_signal'] = np.where(df['SMA_Ratio'] < 1, 1, 0)
    df['SMA_position'] = df['SMA_signal'].diff()
    return df

def compute_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta).clip(lower=0).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_signal'] = np.where(df['RSI'] < 30, 1,
                         np.where(df['RSI'] > 70, -1, 0))
    return df

def compute_bollinger_bands(df, window=20):
    df['BB_MA'] = df['close'].rolling(window=window).mean()
    df['BB_std'] = df['close'].rolling(window=window).std()
    df['BB_upper'] = df['BB_MA'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_MA'] - 2 * df['BB_std']
    df['BB_signal'] = np.where(df['close'] < df['BB_lower'], 1,
                         np.where(df['close'] > df['BB_upper'], -1, 0))
    return df

def compute_macd(df, span_short=12, span_long=26, span_signal=9):
    df['EMA_short'] = df['close'].ewm(span=span_short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=span_long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal_line'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()
    df['MACD_signal'] = np.where(df['MACD'] > df['MACD_signal_line'], 1, -1)
    return df

def compute_obv(df):
    df['price_change'] = df['close'].diff()
    df['direction'] = np.where(df['price_change'] > 0, 1,
                               np.where(df['price_change'] < 0, -1, 0))
    df['OBV'] = (df['volume'] * df['direction']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
    df['OBV_signal'] = np.where(df['OBV'] > df['OBV_MA'], 1, -1)
    return df

def compute_stochastic(df, window=14, smooth_window=3):
    df['lowest_low'] = df['close'].rolling(window=window).min()
    df['highest_high'] = df['close'].rolling(window=window).max()
    df['%K'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['%D'] = df['%K'].rolling(window=smooth_window).mean()
    df['stoch_signal'] = np.where(df['%K'] < 20, 1,
                          np.where(df['%K'] > 80, -1, 0))
    return df

def compute_adx(df, window=14):
    df['TR'] = df['close'].diff().abs()
    df['+DM'] = np.where(df['close'].diff() > 0, df['close'].diff(), 0)
    df['-DM'] = np.where(df['close'].diff() < 0, -df['close'].diff(), 0)
    df['TR_smooth'] = df['TR'].rolling(window=window).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=window).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=window).sum()
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    df['DX'] = 100 * (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=window).mean()
    df['ADX_signal'] = np.where((df['ADX'] > 25) & (df['+DI'] > df['-DI']), 1,
                         np.where((df['ADX'] > 25) & (df['+DI'] < df['-DI']), -1, 0))
    return df

def apply_trading_strategy(df):
    df = df.sort_values(by='date').reset_index(drop=True)
    df = compute_sma_signals(df)
    df = compute_rsi(df)
    df = compute_bollinger_bands(df)
    df = compute_macd(df)
    df = compute_obv(df)
    df = compute_stochastic(df)
    df = compute_adx(df)
    # Combine signals from 7 indicators:
    signal_cols = ['SMA_signal', 'RSI_signal', 'BB_signal', 
                   'MACD_signal', 'OBV_signal', 'stoch_signal', 'ADX_signal']
    for col in signal_cols:
        if col not in df.columns:
            df[col] = 0
    df['combined_signal_score'] = df[signal_cols].sum(axis=1)
    threshold = 3
    df['overall_signal'] = np.where(df['combined_signal_score'] >= threshold, 1,
                             np.where(df['combined_signal_score'] <= -threshold, -1, 0))
    # ATR as a volatility proxy
    df['atr'] = df['close'].diff().abs().rolling(window=14).mean()
    return df

#########################
# SQL INTEGRATION & TECHNICAL SIGNAL STORAGE
#########################

def update_technical_signals(db_path):
    try:
        conn = sqlite3.connect(db_path)
        logging.info("Connected to database for technical signals update.")
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return
    
    price_query = "SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices"
    price_df = pd.read_sql_query(price_query, conn, parse_dates=['date'])
    logging.info("Loaded daily price data from nasdaq_100_daily_prices.")
    technical_records = []
    
    for ticker, group in price_df.groupby('ticker'):
        group = group.sort_values(by='date').reset_index(drop=True)
        group = apply_trading_strategy(group)
        group = group.dropna(subset=['SMA_long', 'RSI', 'MACD', 'atr'])
        for _, row in group.iterrows():
            record = (
                ticker,
                row['date'].strftime('%Y-%m-%d'),
                row['SMA_long'],  # 20-day SMA
                row['RSI'],
                row['MACD'],
                row['atr'],
                int(row['overall_signal'])
            )
            technical_records.append(record)
    
    insert_query = """
        INSERT OR IGNORE INTO technical_signals 
        (ticker, signal_date, sma, rsi, macd, atr, signal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    try:
        cursor = conn.cursor()
        cursor.executemany(insert_query, technical_records)
        conn.commit()
        logging.info(f"Inserted {len(technical_records)} records into technical_signals.")
        print(f"Inserted {len(technical_records)} records into technical_signals.")
    except Exception as e:
        logging.error(f"Error inserting technical signals: {e}")
    finally:
        conn.close()

#########################
# RISK METRIC CALCULATION FUNCTIONS
#########################

def compute_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    if daily_returns.std() == 0:
        return 0
    sharpe = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * sqrt(252)
    return sharpe

def compute_max_drawdown(cumulative_returns):
    cummax = cumulative_returns.cummax()
    drawdown = (cummax - cumulative_returns) / cummax
    return drawdown.max()

#########################
# BENCHMARK & BACKTESTING FUNCTIONS
#########################

def benchmark_stats_from_sql(conn, benchmark_ticker: str, starting_date: str, total_investment_value: float):
    query = """
        SELECT date, close FROM nasdaq_100_daily_prices 
        WHERE ticker = ? AND date >= ?
        ORDER BY date
    """
    bench_df = pd.read_sql_query(query, conn, params=(benchmark_ticker, starting_date), parse_dates=['date'])
    bench_df = bench_df.sort_values(by='date').reset_index(drop=True)
    bench_df['benchmark_returns'] = bench_df['close'].diff()
    bench_df = bench_df.dropna()
    total_stocks = floor(total_investment_value / bench_df['close'].iloc[0])
    benchmark_investment_return = total_stocks * bench_df['benchmark_returns']
    total_benchmark_return = benchmark_investment_return.sum()
    profit_pct = floor((total_benchmark_return / total_investment_value) * 100)
    return total_benchmark_return, profit_pct

def backtest_trading_strategy(db_path, starting_date='2000-01-01', investment_value=100000):
    conn = sqlite3.connect(db_path)
    bench_return, bench_profit_pct = benchmark_stats_from_sql(conn, 'SPY', starting_date, investment_value)
    print(f"Benchmark (SPY) profit: ${bench_return} and profit percentage: {bench_profit_pct}%")
    stocks = pd.read_sql_query("SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices", conn, parse_dates=['date'])
    conn.close()
    
    results = []
    
    for stock_ticker in stocks['ticker'].unique():
        stock = stocks[(stocks['ticker'] == stock_ticker) & (stocks['date'] >= starting_date)].copy()
        if stock.empty:
            continue
        stock = stock.sort_values(by='date').reset_index(drop=True)
        stock = apply_trading_strategy(stock)
        stock = stock.dropna(subset=['SMA_Ratio', 'RSI', 'MACD', 'atr', 'BB_lower', 'BB_upper'])
        if stock.empty:
            continue
        stock = stock.reset_index(drop=True)
        # Build trading positions based on overall_signal
        position = [0] * len(stock)
        for i in range(len(stock)):
            if stock.loc[i, 'overall_signal'] == 1:
                position[i] = 1
            elif stock.loc[i, 'overall_signal'] == -1:
                position[i] = 0
            elif i > 0:
                position[i] = position[i-1]
        stock['trading_position'] = position
        
        # Compute strategy daily returns
        stock['returns'] = stock['close'].diff()
        strategy_returns = []
        for i in range(1, len(stock)):
            strategy_returns.append(stock['returns'].iloc[i] * stock['trading_position'].iloc[i-1])
        strategy_returns_df = pd.DataFrame(strategy_returns, columns=['trading_returns'])
        
        number_of_shares = floor(investment_value / stock['close'].iloc[0])
        trading_investment_ret = strategy_returns_df['trading_returns'] * number_of_shares
        total_investment_ret = round(trading_investment_ret.sum(), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
        benchmark_comparison = profit_percentage - bench_profit_pct
        
        # Compute risk metrics: convert trading returns to daily percentages
        daily_pct = strategy_returns_df['trading_returns'] / investment_value
        sharpe_ratio = compute_sharpe_ratio(daily_pct)
        cumulative_returns = (1 + daily_pct).cumprod() - 1
        max_drawdown = compute_max_drawdown(cumulative_returns)
        
        results.append({
            'Ticker': stock_ticker,
            'Profit Gained': total_investment_ret,
            'Profit Percentage': profit_percentage,
            'Benchmark Profit Comparison': benchmark_comparison,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })
    
    returns_df = pd.DataFrame(results)
    return returns_df

def update_backtesting_results(db_path, results_df, test_date, strategy_name):
    """
    Inserts backtesting results into the SQL table backtesting_results.
    Each row is inserted with test_date, strategy_name (augmented by ticker), total_return, sharpe_ratio, and max_drawdown.
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

#########################
# INTERACTIVE PLOTTING FUNCTION (using Plotly)
#########################

def interactive_plot_strategy(ticker, db_path, starting_date='2000-01-01'):
    """
    Creates an interactive Plotly chart for a given ticker.
    The chart includes:
      - Stock close price.
      - 5-day and 20-day SMAs.
      - Buy and sell signals.
      - SPY benchmark close price for comparison.
    """
    # Fetch stock data from SQL
    conn = sqlite3.connect(db_path)
    query = f"SELECT date, close, volume FROM nasdaq_100_daily_prices WHERE ticker = ? AND date >= ? ORDER BY date"
    stock_df = pd.read_sql_query(query, conn, params=(ticker, starting_date), parse_dates=['date'])
    # Fetch benchmark (SPY) data from SQL
    bench_query = f"SELECT date, close FROM nasdaq_100_daily_prices WHERE ticker = ? AND date >= ? ORDER BY date"
    bench_df = pd.read_sql_query(bench_query, conn, params=('SPY', starting_date), parse_dates=['date'])
    conn.close()
    
    if stock_df.empty or bench_df.empty:
        print(f"Data not found for {ticker} or benchmark.")
        return
    
    stock_df = stock_df.sort_values(by='date').reset_index(drop=True)
    stock_df = apply_trading_strategy(stock_df)
    
    # Create interactive figure with subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f"{ticker} Price & Indicators", "SPY Benchmark"))
    
    # Plot stock close price and SMAs
    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name=f"{ticker} Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['SMA_short'], mode='lines', name="SMA 5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['SMA_long'], mode='lines', name="SMA 20"), row=1, col=1)
    
    # Mark buy and sell signals
    buy_df = stock_df[stock_df['overall_signal'] == 1]
    sell_df = stock_df[stock_df['overall_signal'] == -1]
    fig.add_trace(go.Scatter(x=buy_df['date'], y=buy_df['close'], mode='markers', marker_symbol='triangle-up',
                             marker_color='green', marker_size=10, name="Buy Signal"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_df['date'], y=sell_df['close'], mode='markers', marker_symbol='triangle-down',
                             marker_color='red', marker_size=10, name="Sell Signal"), row=1, col=1)
    
    # Plot benchmark SPY price
    fig.add_trace(go.Scatter(x=bench_df['date'], y=bench_df['close'], mode='lines', name="SPY Close", line=dict(color='orange')), row=2, col=1)
    
    fig.update_layout(title=f"Interactive Trading Strategy for {ticker}",
                      xaxis=dict(rangeselector=dict(buttons=list([
                          dict(count=1, label="1m", step="month", stepmode="backward"),
                          dict(count=6, label="6m", step="month", stepmode="backward"),
                          dict(count=1, label="YTD", step="year", stepmode="todate"),
                          dict(count=1, label="1y", step="year", stepmode="backward"),
                          dict(step="all")
                      ])),
                      rangeslider=dict(visible=True),
                      type="date"),
                      height=700)
    
    fig.show()

#########################
# MAIN EXECUTION
#########################

if __name__ == "__main__":
    db_path = os.path.join("database", "data.db")
    
    # Step 1: Update technical_signals table
    update_technical_signals(db_path)
    
    # Step 2: Backtest the trading strategy and compare with SPY benchmark
    returns_df = backtest_trading_strategy(db_path, starting_date='2000-01-01', investment_value=100000)
    print("Backtesting Results:")
    print(returns_df)
    
    # Step 3: Update backtesting_results table in SQL with risk metrics
    test_date = datetime.now().strftime('%Y-%m-%d')
    strategy_name = "Combined Technical Strategy"
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
    
    # Step 5: Rank the top 25 stocks by Profit Gained and print sector counts
    ranked_df = returns_df.sort_values(by='Profit Gained', ascending=False).head(25)
    print("\nTop 25 Stocks:")
    print(ranked_df)
    print("\nTop 25 Performing Sectors:")
    print(ranked_df['Sector'].value_counts())
    print("\nTotal Initial Sector Counts:")
    print(returns_df['Sector'].value_counts())
    
    # Step 6: Generate an interactive plot for a specific ticker (e.g., AAPL)
    interactive_plot_strategy('AAPL', db_path, starting_date='2000-01-01')
