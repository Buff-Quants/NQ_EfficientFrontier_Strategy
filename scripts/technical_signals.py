import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from math import floor
import yfinance as yf  # Used only for sector info

# Set up logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

##############################################
# Technical Indicator Functions – using 'close'
##############################################

def compute_sma_signals(df, short_window=5, long_window=20):
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    df['SMA_Ratio'] = df['SMA_long'] / df['SMA_short']
    return df

def compute_volume_sma_ratio(df, short_window=5, long_window=20):
    df['vol_SMA_short'] = df['volume'].rolling(window=short_window).mean()
    df['vol_SMA_long'] = df['volume'].rolling(window=long_window).mean()
    df['SMA_Volume_Ratio'] = df['vol_SMA_long'] / df['vol_SMA_short']
    return df

def compute_atr_and_ratio(df, window=14):
    df['ATR'] = df['close'].diff().abs().rolling(window=window).mean()
    df['ATR_Ratio'] = df['ATR'] / df['ATR'].shift(1)
    return df

def compute_rsi(df, window=20):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI_20'] = 100 - (100 / (1 + rs))
    return df

def compute_bollinger_bands(df, window=20):
    df['BB_MA'] = df['close'].rolling(window=window).mean()
    df['BB_std'] = df['close'].rolling(window=window).std()
    df['upperband'] = df['BB_MA'] + 2 * df['BB_std']
    df['lowerband'] = df['BB_MA'] - 2 * df['BB_std']
    return df

def compute_macd(df, span_short=12, span_long=26, span_signal=9):
    df['EMA_short'] = df['close'].ewm(span=span_short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=span_long, adjust=False).mean()
    df['MACD_Value'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD_Value'].ewm(span=span_signal, adjust=False).mean()
    return df

def compute_stochastic(df, window=20, smooth_window=3):
    df['lowest_low'] = df['close'].rolling(window=window).min()
    df['highest_high'] = df['close'].rolling(window=window).max()
    df['20Day_%K'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['20Day_%D'] = df['20Day_%K'].rolling(window=smooth_window).mean()
    return df

def compute_all_indicators(df):
    df = df.sort_values(by='date').reset_index(drop=True)
    df = compute_sma_signals(df)
    df = compute_volume_sma_ratio(df)
    df = compute_atr_and_ratio(df)
    df = compute_rsi(df)
    df = compute_bollinger_bands(df)
    df = compute_macd(df)
    df = compute_stochastic(df)
    return df

##############################################
# Trading Strategy Function
##############################################

def trading_strategy(prices, SMA_Ratio, SMA_Volume_Ratio, ATR_Ratio,
                     Day20_K, Day20_D, RSI_20, MACD_Value, MACD_Signal,
                     upperband, lowerband):
    buy_price = []
    sell_price = []
    trading_signal = []
    signal = 0  # 0 = no position, 1 = buy/long, -1 = sell/exit
    for i in range(len(prices)):
        # All BUY conditions must be true to buy:
        if (SMA_Ratio[i] < 1) and (SMA_Volume_Ratio[i] < 1) and (ATR_Ratio[i] < 1) and \
           (Day20_K[i] < 30) and (Day20_D[i] < 30) and (RSI_20[i] < 30) and \
           (MACD_Value[i] > MACD_Signal[i]) and (prices[i] < lowerband[i]):
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                trading_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                trading_signal.append(0)
        # Any one SELL condition triggers a sell:
        elif (SMA_Ratio[i] > 1) or (SMA_Volume_Ratio[i] > 1) or (ATR_Ratio[i] > 1) or \
             (Day20_K[i] > 70) or (Day20_D[i] > 70) or (RSI_20[i] > 70) or \
             (MACD_Value[i] < MACD_Signal[i]) or (prices[i] > upperband[i]):
            if signal != -1 and signal != 0:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                trading_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                trading_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            trading_signal.append(0)
    return buy_price, sell_price, trading_signal

##############################################
# Database Update – Insert Technical Signals
##############################################

def update_technical_signals(db_path):
    try:
        conn = sqlite3.connect(db_path)
        logging.info("Connected to database for technical signals update.")
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return
    
    # Read price data from SQL; columns: ticker, date, close, volume
    price_query = "SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices"
    price_df = pd.read_sql_query(price_query, conn, parse_dates=['date'])
    logging.info("Loaded daily price data from nasdaq_100_daily_prices.")
    
    technical_records = []
    for ticker, group in price_df.groupby('ticker'):
        group = group.sort_values(by='date').reset_index(drop=True)
        group = compute_all_indicators(group)
        # Insert only rows with complete indicator data
        group = group.dropna(subset=['SMA_Ratio', 'SMA_Volume_Ratio', 'ATR_Ratio', '20Day_%K', '20Day_%D', 'RSI_20', 'MACD_Value'])
        for _, row in group.iterrows():
            # For technical_signals, we store: ticker, signal_date, sma (20-day), rsi (RSI_20), macd (MACD_Value), atr, and overall signal.
            overall_signal = 0  # (if desired, you can store the strategy’s overall signal later)
            record = (
                ticker,
                row['date'].strftime('%Y-%m-%d'),
                row['SMA_long'],
                row['RSI_20'],
                row['MACD_Value'],
                row['ATR'],
                overall_signal
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

##############################################
# Benchmark & Backtesting – using SQL benchmark data
##############################################

def benchmark_stats_from_sql(conn, benchmark_ticker: str, starting_date: str, total_investment_value: float):
    """
    Query benchmark data (e.g., SPY) from the SQL table and compute benchmark returns.
    """
    query = f"""
        SELECT date, close FROM nasdaq_100_daily_prices 
        WHERE ticker = ? AND date >= ?
        ORDER BY date
    """
    bench_df = pd.read_sql_query(query, conn, params=(benchmark_ticker, starting_date), parse_dates=['date'])
    bench_df = bench_df.sort_values(by='date').reset_index(drop=True)
    # Compute daily returns (difference)
    bench_df['benchmark_returns'] = bench_df['close'].diff()
    bench_df = bench_df.dropna()
    total_stocks = floor(total_investment_value / bench_df['close'].iloc[0])
    benchmark_investment_return = total_stocks * bench_df['benchmark_returns']
    return benchmark_investment_return.sum(), floor((total_stocks * bench_df['benchmark_returns'].sum() / total_investment_value) * 100)

def backtest_trading_strategy(db_path, starting_date='2000-01-01', investment_value=100000):
    """
    Backtest the trading strategy for all NASDAQ 100 stocks (from SQL) and compare performance
    to benchmark indexes (e.g. SPY) also fetched from SQL.
    """
    conn = sqlite3.connect(db_path)
    # Compute benchmark stats from SQL for SPY (you could also do for '^NDX' if desired)
    bench_return, bench_profit_pct = benchmark_stats_from_sql(conn, 'SPY', starting_date, investment_value)
    print(f"Benchmark (SPY) profit: ${bench_return} and profit percentage: {bench_profit_pct}%")
    
    # Load price data for NASDAQ-100 stocks from SQL
    stocks = pd.read_sql_query("SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices", conn, parse_dates=['date'])
    conn.close()
    
    returns_df = pd.DataFrame(columns=['Ticker', 'Profit Gained', 'Profit Percentage', 'Benchmark Profit Comparison'])
    
    for stock_ticker in stocks['ticker'].unique():
        stock = stocks[(stocks['ticker'] == stock_ticker) & (stocks['date'] >= starting_date)].copy()
        if stock.empty:
            continue
        stock = stock.sort_values(by='date').reset_index(drop=True)
        stock = compute_all_indicators(stock)
        # Drop rows with missing data for required indicators
        stock = stock.dropna(subset=['SMA_Ratio', 'SMA_Volume_Ratio', 'ATR_Ratio', '20Day_%K', '20Day_%D', 'RSI_20', 'MACD_Value', 'MACD_Signal', 'upperband', 'lowerband'])
        if stock.empty:
            continue
        
        # Create Trading Strategy signals
        buy_price, sell_price, trading_signal = trading_strategy(
            prices = stock['close'].values,
            SMA_Ratio = stock['SMA_Ratio'].values,
            SMA_Volume_Ratio = stock['SMA_Volume_Ratio'].values,
            ATR_Ratio = stock['ATR_Ratio'].values,
            Day20_K = stock['20Day_%K'].values,
            Day20_D = stock['20Day_%D'].values,
            RSI_20 = stock['RSI_20'].values,
            MACD_Value = stock['MACD_Value'].values,
            MACD_Signal = stock['MACD_Signal'].values,
            upperband = stock['upperband'].values,
            lowerband = stock['lowerband'].values
        )
        stock['buy_price'] = buy_price
        stock['sell_price'] = sell_price
        stock['trading_signal'] = trading_signal

        # Create Trading Position based on signals:
        position = [0] * len(stock)
        for i in range(len(stock)):
            if trading_signal[i] == 1:
                position[i] = 1
            elif trading_signal[i] == -1:
                position[i] = 0
            elif i > 0:
                position[i] = position[i-1]
        stock['trading_position'] = position

        # Backtest: calculate daily returns and strategy returns
        stock['returns'] = stock['close'].diff()
        trading_strategy_ret = []
        for i in range(1, len(stock)):
            trading_strategy_ret.append(stock['returns'].iloc[i] * stock['trading_position'].iloc[i-1])
        trading_strategy_ret_df = pd.DataFrame(trading_strategy_ret, columns=['trading_returns'])
        
        number_of_stocks = floor(investment_value / stock['close'].iloc[0])
        trading_investment_ret = trading_strategy_ret_df['trading_returns'] * number_of_stocks
        total_investment_ret = round(trading_investment_ret.sum(), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
        benchmark_comparison = profit_percentage - bench_profit_pct
        
        returns_df = returns_df.append({
            'Ticker': stock_ticker,
            'Profit Gained': total_investment_ret,
            'Profit Percentage': profit_percentage,
            'Benchmark Profit Comparison': benchmark_comparison
        }, ignore_index=True)
    
    return returns_df

##############################################
# Main Execution
##############################################

if __name__ == "__main__":
    db_path = os.path.join("database", "data.db")
    
    # Step 1: Update technical_signals table with computed technical values.
    update_technical_signals(db_path)
    
    # Step 2: Backtest the trading strategy for all stocks using SQL benchmark data.
    returns_df = backtest_trading_strategy(db_path, starting_date='2000-01-01', investment_value=100000)
    print("Backtesting Results:")
    print(returns_df)
    
    # Step 3: Merge Sector Information using yfinance.
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
    
    # Step 4: Rank top 25 stocks by Profit Gained.
    ranked_df = returns_df.sort_values(by='Profit Gained', ascending=False).head(25)
    print("\nTop 25 Stocks:")
    print(ranked_df)
    
    # Print sector counts for top 25 and overall
    print("\nTop 25 Performing Sectors:")
    print(ranked_df['Sector'].value_counts())
    print("\nTotal Initial Sector Counts:")
    print(returns_df['Sector'].value_counts())
    
    # Additional portfolio optimization steps can be added here...
