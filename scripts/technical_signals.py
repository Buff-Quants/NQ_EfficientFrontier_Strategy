## NEED TO PUT SPY DATA INTO SQL BEFORE THIS WORKS ##
# ALSO WOULD LIKE TO COMPARE TO NASDAQ-100 FUTURES ## 

'''
THIS SCRIPT COMPUTES 7 TEHCNICAL INDICATORS, USES ALL 7 TO GENERATE BUY SIGNALS, WHEREAS ONLY 1 SELL SIGNAL IS NEEDED TO SELL 
THE STRATEGY IS THEN BACKTESTED ON NASDAQ-100 STOCKS, AND THE TOP 25 STOCKS ARE RANKED BY PROFIT GAINED 
THE SECTOR OF EACH STOCK IS THEN DETERMINED USING YFINANCE 
THE TOP 25 STOCKS ARE THEN PRINTED ALONG WITH THE SECTOR COUNTS FOR THE TOP 25 AND OVERALL 
THE SECTOR COUNTS ARE PRINTED TO SHOW WHICH SECTORS ARE PERFORMING WELL 
'''

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from math import floor
import yfinance as yf

# Set up logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

##############################################
# Indicator Functions – using 'close' price
##############################################

def compute_sma_signals(df, short_window=5, long_window=20):
    """
    Compute 5-day and 20-day SMAs and their ratio.
    """
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    # If SMA_long/SMA_short < 1, then short-term SMA is above long-term SMA.
    df['SMA_Ratio'] = df['SMA_long'] / df['SMA_short']
    return df

def compute_volume_sma_ratio(df, short_window=5, long_window=20):
    """
    Compute short-term and long-term volume moving averages and their ratio.
    """
    df['vol_SMA_short'] = df['volume'].rolling(window=short_window).mean()
    df['vol_SMA_long'] = df['volume'].rolling(window=long_window).mean()
    # Ratio: if vol_SMA_long / vol_SMA_short < 1, then volume is trending lower.
    df['SMA_Volume_Ratio'] = df['vol_SMA_long'] / df['vol_SMA_short']
    return df

def compute_atr_and_ratio(df, window=14):
    """
    Compute the ATR as the rolling mean of the true range (using absolute daily change)
    and then compute ATR_Ratio as ATR / ATR.shift(1).
    """
    df['ATR'] = df['close'].diff().abs().rolling(window=window).mean()
    df['ATR_Ratio'] = df['ATR'] / df['ATR'].shift(1)
    return df

def compute_rsi(df, window=20):
    """
    Compute RSI with the specified window (here 20-day for RSI_20).
    """
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI_20'] = 100 - (100 / (1 + rs))
    return df

def compute_bollinger_bands(df, window=20):
    """
    Compute Bollinger Bands using a 20-day window.
    """
    df['BB_MA'] = df['close'].rolling(window=window).mean()
    df['BB_std'] = df['close'].rolling(window=window).std()
    df['upperband'] = df['BB_MA'] + 2 * df['BB_std']
    df['lowerband'] = df['BB_MA'] - 2 * df['BB_std']
    return df

def compute_macd(df, span_short=12, span_long=26, span_signal=9):
    """
    Compute MACD and its signal line.
    """
    df['EMA_short'] = df['close'].ewm(span=span_short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=span_long, adjust=False).mean()
    df['MACD_Value'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD_Value'].ewm(span=span_signal, adjust=False).mean()
    return df

def compute_stochastic(df, window=20, smooth_window=3):
    """
    Compute the stochastic oscillator (%K and %D) using a 20-day window.
    """
    df['lowest_low'] = df['close'].rolling(window=window).min()
    df['highest_high'] = df['close'].rolling(window=window).max()
    df['20Day_%K'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['20Day_%D'] = df['20Day_%K'].rolling(window=smooth_window).mean()
    return df

def compute_all_indicators(df):
    """
    Compute all required technical indicators and add them as columns.
    """
    df = df.sort_values(by='date').reset_index(drop=True)
    df = compute_sma_signals(df)
    df = compute_volume_sma_ratio(df)
    df = compute_atr_and_ratio(df)
    df = compute_rsi(df)         # RSI_20
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
    """
    For each time point:
      - All BUY conditions must be true to buy.
      - If any SELL condition is met, then sell.
    BUY conditions:
      SMA_Ratio < 1,
      SMA_Volume_Ratio < 1,
      ATR_Ratio < 1,
      20Day_%K < 30,
      20Day_%D < 30,
      RSI_20 < 30,
      MACD_Value > MACD_Signal,
      price < lowerband.
    SELL conditions:
      Any one of:
      SMA_Ratio > 1,
      SMA_Volume_Ratio > 1,
      ATR_Ratio > 1,
      20Day_%K > 70,
      20Day_%D > 70,
      RSI_20 > 70,
      MACD_Value < MACD_Signal,
      price > upperband.
    """
    buy_price = []
    sell_price = []
    trading_signal = []
    signal = 0  # 0 = no position, 1 = long, -1 = sold/exit
    for i in range(len(prices)):
        # Buy: all buy conditions must be met
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
        # Sell: if any sell condition is met
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
# Database Update – Inserting Technical Signals
##############################################

def update_technical_signals(db_path):
    """
    Reads price data from the SQL table, computes technical indicators,
    and inserts calculated technical values into the technical_signals table.
    """
    try:
        conn = sqlite3.connect(db_path)
        logging.info("Connected to database for technical signals update.")
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return
    
    # Read price data: ticker, date, close, volume
    price_query = "SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices"
    price_df = pd.read_sql_query(price_query, conn, parse_dates=['date'])
    logging.info("Loaded daily price data from nasdaq_100_daily_prices.")
    
    technical_records = []
    
    # Process each ticker individually
    for ticker, group in price_df.groupby('ticker'):
        group = group.sort_values(by='date').reset_index(drop=True)
        group = compute_all_indicators(group)
        # Insert only rows where we have full indicator data (drop initial NaNs)
        group = group.dropna(subset=['SMA_long', 'RSI_20', 'MACD_Value', 'ATR', '20Day_%K'])
        for _, row in group.iterrows():
            # For technical_signals table, we store: ticker, signal_date, sma (20-day SMA), rsi (RSI_20),
            # macd (MACD_Value), atr, and overall signal (we use a simple overall_signal based on our trading strategy)
            # Here, we’ll also store our computed indicator values if needed.
            # For overall signal, we can store 1 for buy, -1 for sell, 0 otherwise.
            overall_signal = int(row.get('overall_signal', 0))
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
# Backtesting & Trading Strategy Statistics
##############################################

def benchmark_stats(starting_date, total_investment_value, spy_df):
    """
    Given SPY data (from SPY.csv), compute benchmark returns.
    """
    SPY = spy_df['Adj Close']
    benchmark = pd.DataFrame(np.diff(SPY)).rename(columns = {0:'benchmark_returns'})
    total_stocks = floor(total_investment_value / SPY.iloc[0])
    benchmark_investment_return = []
    for i in range(len(benchmark['benchmark_returns'])):
        returns = total_stocks * benchmark['benchmark_returns'].iloc[i]
        benchmark_investment_return.append(returns)
    benchmark_investment_return_df = pd.DataFrame(benchmark_investment_return).rename(columns = {0:'investment_returns'})
    return benchmark_investment_return_df

def backtest_trading_strategy(db_path, spy_csv_path):
    """
    For each NASDAQ 100 stock (from the SQL daily prices), compute the trading strategy
    and compare its performance to a SPY benchmark.
    Returns a DataFrame with profit stats.
    """
    # Load SPY data for benchmark stats
    df_spy = pd.read_csv(spy_csv_path)
    benchmark = benchmark_stats('2000-01-01', 100000, df_spy)
    total_benchmark_investment_returns = round(sum(benchmark['investment_returns']), 2)
    benchmark_profit_percentage = floor((total_benchmark_investment_returns / 100000) * 100)
    print(f"Benchmark Stats from date 2000-01-01 to present using $100000 investment:")
    print(f"Benchmark profit dollar amount: ${total_benchmark_investment_returns}")
    print(f"Benchmark profit percentage: {benchmark_profit_percentage}%")
    print()

    # Load price data from SQL
    conn = sqlite3.connect(db_path)
    stocks = pd.read_sql_query("SELECT ticker, date, close, volume FROM nasdaq_100_daily_prices", conn, parse_dates=['date'])
    conn.close()
    
    returns_df = pd.DataFrame(columns=['Ticker', 'Profit Gained', 'Profit Percentage', 'Benchmark Profit Comparison'])
    
    for stock_ticker in stocks['ticker'].unique():
        stock = stocks[(stocks['ticker'] == stock_ticker) & (stocks['date'] >= '2000-01-01')].copy()
        if stock.empty:
            continue
        stock = stock.sort_values(by='date').reset_index(drop=True)
        stock = compute_all_indicators(stock)
        # Drop rows with insufficient indicator data
        stock = stock.dropna(subset=['SMA_Ratio', 'SMA_Volume_Ratio', 'ATR_Ratio', '20Day_%K', '20Day_%D', 'RSI_20', 'MACD_Value', 'MACD_Signal', 'upperband', 'lowerband'])
        
        # Skip if after dropping there is not enough data
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

        # Create Trading Position:
        # Initialize position as a list; assume first position is neutral (0) then update based on signals.
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
        
        # Compute investment return given an initial investment of $100000
        investment_value = 100000
        number_of_stocks = floor(investment_value / stock['close'].iloc[0])
        trading_investment_ret = []
        for ret in trading_strategy_ret_df['trading_returns']:
            trading_investment_ret.append(number_of_stocks * ret)
        total_investment_ret = round(sum(trading_investment_ret), 2)
        profit_percentage = floor((total_investment_ret / investment_value) * 100)
        benchmark_comparison = profit_percentage - benchmark_profit_percentage
        
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
    # Define database path and SPY csv path
    db_path = os.path.join("database", "data.db")
    spy_csv_path = "SPY.csv"  # Ensure SPY.csv is in your working directory
    
    # Step 1: Update technical_signals table with computed technical values.
    update_technical_signals(db_path)
    
    # Step 2: Backtest the trading strategy for all NASDAQ 100 stocks.
    returns_df = backtest_trading_strategy(db_path, spy_csv_path)
    
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
