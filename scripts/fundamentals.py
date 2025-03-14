#!/usr/bin/env python3
"""
Fundamental analysis module for quantitative investment strategy.
Implements CAPM and Fama-French models for expected returns estimation.
"""

import os
import sys
import sqlite3
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import statsmodels.api as sm
from typing import Dict, Tuple, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fundamentals.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Cache for downloaded data
_CACHE = {
    'market_data': {},
    'risk_free_data': {},
    'ff_factors': {}
}

def get_price_data(conn, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data for a ticker from the database."""
    try:
        query = f"""
        SELECT date, close, volume
        FROM nasdaq_100_daily_prices
        WHERE ticker = '{ticker}'
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        logging.info(f"Executing SQL: {query}")
        df = pd.read_sql_query(query, conn)
        
        logging.info(f"Retrieved {len(df)} rows of price data for {ticker}")
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert date to datetime and handle duplicates
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates('date', keep='last')
        df.set_index('date', inplace=True)
        
        # Rename columns to match yfinance format
        df.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Error getting price data for {ticker}: {e}")
        return pd.DataFrame()

def get_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Get market data (S&P 500 as a proxy) with caching."""
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _CACHE['market_data']:
        return _CACHE['market_data'][cache_key]
    
    try:
        logging.info(f"Downloading market data from {start_date} to {end_date}")
        market_data = yf.download('^GSPC', start=start_date, end=end_date)
        
        if not market_data.empty:
            _CACHE['market_data'][cache_key] = market_data
            
        return market_data
    except Exception as e:
        logging.error(f"Error getting market data: {e}")
        return pd.DataFrame()

def get_risk_free_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """Get risk-free rate data (13-week Treasury bill) with caching."""
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _CACHE['risk_free_data']:
        return _CACHE['risk_free_data'][cache_key]
    
    try:
        adjusted_end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        logging.info(f"Downloading risk-free rate data")
        rf_data = yf.download('^IRX', start=start_date, end=adjusted_end_date)
        
        if rf_data.empty:
            return pd.DataFrame()
            
        # Convert annual rate to daily rate
        if isinstance(rf_data.columns, pd.MultiIndex):
            close_values = rf_data.loc[:, ('Close', slice(None))].iloc[:, 0]
            rf_daily = close_values / 100 / 252
            rf_data = pd.DataFrame({'daily_rate': rf_daily})
        else:
            rf_data['daily_rate'] = rf_data['Close'] / 100 / 252
        
        if not rf_data.empty:
            _CACHE['risk_free_data'][cache_key] = rf_data
            
        return rf_data
    except Exception as e:
        logging.error(f"Error getting risk-free rate: {e}")
        return pd.DataFrame()

def get_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """Get Fama-French factors using ETFs as proxies with caching."""
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _CACHE['ff_factors']:
        return _CACHE['ff_factors'][cache_key]
    
    try:
        # Use yfinance to get French data (using ETFs as proxies)
        logging.info(f"Downloading Fama-French factor proxies")
        
        # Get ETF data for proxies
        iwm = yf.download('IWM', start=start_date, end=end_date)['Close']  # Small cap
        spy = yf.download('SPY', start=start_date, end=end_date)['Close']  # Large cap
        iwd = yf.download('IWD', start=start_date, end=end_date)['Close']  # Value
        iwf = yf.download('IWF', start=start_date, end=end_date)['Close']  # Growth
        
        # Calculate returns and factors
        iwm_ret = iwm.pct_change().dropna()
        spy_ret = spy.pct_change().dropna()
        iwd_ret = iwd.pct_change().dropna()
        iwf_ret = iwf.pct_change().dropna()
        
        # Extract Series values if they're DataFrames
        if isinstance(iwm_ret, pd.DataFrame): iwm_ret = iwm_ret.iloc[:, 0]
        if isinstance(spy_ret, pd.DataFrame): spy_ret = spy_ret.iloc[:, 0]
        if isinstance(iwd_ret, pd.DataFrame): iwd_ret = iwd_ret.iloc[:, 0]
        if isinstance(iwf_ret, pd.DataFrame): iwf_ret = iwf_ret.iloc[:, 0]
            
        # Size factor (SMB) and Value factor (HML)
        smb_proxy = iwm_ret - spy_ret
        hml_proxy = iwd_ret - iwf_ret
        
        # Get risk-free rate
        rf_data = get_risk_free_rate(start_date, end_date)
        
        # Combine factors
        factors = pd.DataFrame({
            'SMB': smb_proxy,
            'HML': hml_proxy
        })
        
        # Add risk-free rate if available
        if 'daily_rate' in rf_data.columns:
            rf_series = rf_data['daily_rate']
            factors = factors.join(rf_series, how='left', rsuffix='_rf')
            factors.rename(columns={'daily_rate': 'RF'}, inplace=True)
        
        # Drop NaN values
        factors = factors.dropna()
        
        if not factors.empty:
            _CACHE['ff_factors'][cache_key] = factors
            
        return factors
    except Exception as e:
        logging.error(f"Error getting Fama-French factors: {e}")
        return pd.DataFrame()

def calculate_returns(prices: pd.DataFrame) -> pd.Series:
    """Calculate daily returns from price data."""
    try:
        if isinstance(prices.columns, pd.MultiIndex):
            close_values = prices.loc[:, ('Close', slice(None))].iloc[:, 0]
            returns = close_values.pct_change().dropna()
            return returns
        elif 'Close' in prices.columns:
            returns = prices['Close'].pct_change().dropna()
            return returns
        elif 'close' in prices.columns:
            returns = prices['close'].pct_change().dropna()
            return returns
        else:
            logging.warning("No close/Close column found in price data")
            return pd.Series()
    except Exception as e:
        logging.error(f"Error calculating returns: {e}")
        return pd.Series()

def validate_data(stock_returns, market_returns, risk_free_rate):
    """Validate and align data for analysis."""
    # Convert to Series if DataFrame
    if isinstance(stock_returns, pd.DataFrame):
        if 'Close' in stock_returns.columns:
            stock_returns = stock_returns['Close']
        else:
            stock_returns = stock_returns.iloc[:, 0]
    
    if isinstance(market_returns, pd.DataFrame):
        if isinstance(market_returns.columns, pd.MultiIndex):
            market_returns = market_returns[('Close', market_returns.columns.get_level_values(1)[0])]
        elif 'Close' in market_returns.columns:
            market_returns = market_returns['Close']
        else:
            market_returns = market_returns.iloc[:, 0]
    
    if isinstance(risk_free_rate, pd.DataFrame):
        risk_free_rate = risk_free_rate.iloc[:, 0]
    
    # Drop NaN values
    stock_returns = stock_returns.dropna()
    market_returns = market_returns.dropna()
    risk_free_rate = risk_free_rate.dropna()
    
    # Align data on dates
    data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns,
        'rf': risk_free_rate
    })
    data = data.dropna()
    
    return data['stock'], data['market'], data['rf']

def calculate_beta_capm(stock_excess_returns, market_excess_returns):
    """Calculate beta using regression for CAPM."""
    try:
        # Create a DataFrame for alignment
        data = pd.DataFrame({
            'stock': stock_excess_returns,
            'market': market_excess_returns
        }).dropna()
        
        if len(data) < 30:
            return np.nan, np.nan, np.nan
            
        X = data['market'].values.reshape(-1, 1)
        y = data['stock'].values
        
        model = sm.OLS(y, sm.add_constant(X)).fit()
        beta = model.params[1]
        alpha = model.params[0]
        r_squared = model.rsquared
        
        return beta, alpha, r_squared
    except Exception as e:
        logging.error(f"Error calculating beta: {e}")
        return np.nan, np.nan, np.nan

def calculate_fama_french_factors(stock_excess_returns, factors_data):
    """Calculate Fama-French factor loadings."""
    try:
        # Create a DataFrame with stock excess returns
        data = pd.DataFrame({'excess_return': stock_excess_returns})
        
        # Join with factors data
        data = data.join(factors_data).dropna()
        
        # Prepare X and y for regression
        X_cols = ['SMB', 'HML']
        if 'MKT-RF' in data.columns:
            X_cols = ['MKT-RF'] + X_cols
        
        X = data[X_cols]
        y = data['excess_return']
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Extract coefficients
        coefficients = {col: model.params[col] for col in X_cols}
        
        # Return factor loadings
        result = {
            'alpha': model.params['const'],
            'r_squared': model.rsquared
        }
        
        # Add factor betas
        if 'MKT-RF' in coefficients:
            result['market_beta'] = coefficients['MKT-RF']
        result['smb_beta'] = coefficients['SMB']
        result['hml_beta'] = coefficients['HML']
        
        return result
    except Exception as e:
        logging.error(f"Error in Fama-French regression: {e}")
        return {
            'alpha': np.nan,
            'market_beta': np.nan,
            'smb_beta': np.nan,
            'hml_beta': np.nan,
            'r_squared': np.nan
        }

def calculate_expected_returns_capm(beta, risk_free_rate, market_premium):
    """Calculate expected returns using CAPM."""
    try:
        return risk_free_rate + beta * market_premium
    except Exception as e:
        logging.error(f"Error calculating CAPM expected returns: {e}")
        return np.nan

def calculate_expected_returns_ff(factor_loadings, factor_premiums, risk_free_rate):
    """Calculate expected returns using Fama-French model."""
    try:
        # Extract betas and premiums
        market_beta = factor_loadings.get('market_beta', 0.0)
        smb_beta = factor_loadings.get('smb_beta', 0.0)
        hml_beta = factor_loadings.get('hml_beta', 0.0)
        
        market_premium = factor_premiums.get('MKT-RF', 0.0)
        smb_premium = factor_premiums.get('SMB', 0.0)
        hml_premium = factor_premiums.get('HML', 0.0)
        
        # Calculate expected return
        return risk_free_rate + market_beta * market_premium + smb_beta * smb_premium + hml_beta * hml_premium
    except Exception as e:
        logging.error(f"Error calculating FF expected returns: {e}")
        return np.nan

def calculate_sharpe_ratio(excess_returns):
    """Calculate Sharpe ratio."""
    try:
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {e}")
        return np.nan

def run_fundamental_analysis(conn, ticker: str, start_date: str, end_date: str, model: str = 'capm') -> Optional[Dict]:
    """Run fundamental analysis for a ticker."""
    try:
        logging.info(f"Running {model.upper()} analysis for {ticker} from {start_date} to {end_date}")
        
        # Get data
        stock_data = get_price_data(conn, ticker, start_date, end_date)
        if stock_data.empty:
            logging.warning(f"No price data available for {ticker}")
            return None
            
        market_data = get_market_data(start_date, end_date)
        if market_data.empty:
            logging.warning(f"No market data available for analysis period")
            return None
            
        risk_free_data = get_risk_free_rate(start_date, end_date)
        if risk_free_data.empty or 'daily_rate' not in risk_free_data.columns:
            logging.warning(f"No risk-free rate data available for analysis period")
            return None
        
        # Calculate returns
        stock_returns = calculate_returns(stock_data)
        market_returns = calculate_returns(market_data)
        risk_free_rate = risk_free_data['daily_rate']
        
        # Validate and align data
        stock_returns, market_returns, risk_free_rate = validate_data(
            stock_returns, market_returns, risk_free_rate
        )
        
        if len(stock_returns) < 30:  # Require at least 30 days of data
            logging.warning(f"Insufficient data for {ticker}: only {len(stock_returns)} days")
            return None
            
        # Calculate excess returns
        stock_excess_returns = stock_returns - risk_free_rate
        market_excess_returns = market_returns - risk_free_rate
        
        # Calculate average rates and market premium
        avg_rf_rate = risk_free_rate.mean()
        market_premium = market_excess_returns.mean()
        
        # Initialize results dictionary
        result = {
            'ticker': ticker,
            'date': end_date,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model,
            'market_premium': market_premium * 252,  # Annualized market premium
            'risk_free_rate': avg_rf_rate * 252  # Annualized risk-free rate
        }
        
        # Run CAPM analysis
        beta, alpha, r_squared = calculate_beta_capm(stock_excess_returns, market_excess_returns)
        expected_return_capm = calculate_expected_returns_capm(beta, avg_rf_rate, market_premium)
        
        result.update({
            'beta': beta,
            'alpha': alpha * 252,  # Annualize
            'r_squared': r_squared,
            'expected_return_capm': expected_return_capm * 252  # Annualize
        })
        
        # Run Fama-French analysis if requested
        if model.lower() == 'fama_french':
            ff_factors = get_fama_french_factors(start_date, end_date)
            
            if not ff_factors.empty:
                # Add market excess returns to factors
                ff_factors['MKT-RF'] = market_excess_returns
                
                # Calculate factor loadings
                ff_loadings = calculate_fama_french_factors(stock_excess_returns, ff_factors)
                
                # Calculate factor premiums
                factor_premiums = {
                    'MKT-RF': market_premium * 252,  # Annualized
                    'SMB': ff_factors['SMB'].mean() * 252,  # Annualized
                    'HML': ff_factors['HML'].mean() * 252   # Annualized
                }
                
                # Calculate expected return
                expected_return_ff = calculate_expected_returns_ff(
                    ff_loadings, 
                    {k: v/252 for k, v in factor_premiums.items()},  # De-annualize for calculation
                    avg_rf_rate
                )
                
                # Add Fama-French results
                result.update({
                    'market_beta': ff_loadings.get('market_beta', beta),  # Use CAPM beta if FF market beta not available
                    'smb_beta': ff_loadings['smb_beta'],
                    'hml_beta': ff_loadings['hml_beta'],
                    'ff_alpha': ff_loadings['alpha'] * 252,  # Annualize
                    'ff_r_squared': ff_loadings['r_squared'],
                    'expected_return_ff': expected_return_ff * 252,  # Annualize
                    'smb_premium': factor_premiums['SMB'],
                    'hml_premium': factor_premiums['HML']
                })
        
        # Calculate Sharpe ratio
        result['sharpe_ratio'] = calculate_sharpe_ratio(stock_excess_returns)
        
        return result
    except Exception as e:
        logging.error(f"Error in fundamental analysis for {ticker}: {e}")
        return None

def store_fundamental_results(conn, result: Dict) -> bool:
    """Store fundamental analysis results in the database."""
    try:
        cursor = conn.cursor()
        
        # Check if entry already exists for this ticker and date
        if result['model'].lower() == 'capm':
            table_name = 'fundamental_analysis_capm'
        else:  # Fama-French model
            table_name = 'fundamental_analysis_ff'
            
        cursor.execute(f"""
        SELECT id FROM {table_name}
        WHERE ticker = ? AND date = ?
        """, (result['ticker'], result['date']))
        
        existing_entry = cursor.fetchone()
        
        if existing_entry:
            # Update existing entry
            if result['model'].lower() == 'capm':
                cursor.execute("""
                UPDATE fundamental_analysis_capm
                SET beta = ?, alpha = ?, expected_return = ?, 
                    sharpe_ratio = ?, r_squared = ?, analysis_date = ?,
                    market_premium = ?, risk_free_rate = ?
                WHERE ticker = ? AND date = ?
                """, (
                    result['beta'], result['alpha'], result['expected_return_capm'],
                    result['sharpe_ratio'], result['r_squared'], result['analysis_date'],
                    result['market_premium'], result['risk_free_rate'],
                    result['ticker'], result['date']
                ))
            else:  # Fama-French model
                cursor.execute("""
                UPDATE fundamental_analysis_ff
                SET beta = ?, alpha = ?, smb_beta = ?, hml_beta = ?, ff_alpha = ?,
                    expected_return_capm = ?, expected_return_ff = ?, sharpe_ratio = ?, 
                    r_squared = ?, ff_r_squared = ?, smb_premium = ?, hml_premium = ?,
                    market_premium = ?, analysis_date = ?
                WHERE ticker = ? AND date = ?
                """, (
                    result['beta'], result['alpha'], result.get('smb_beta', np.nan),
                    result.get('hml_beta', np.nan), result.get('ff_alpha', np.nan),
                    result['expected_return_capm'], result.get('expected_return_ff', np.nan),
                    result['sharpe_ratio'], result['r_squared'],
                    result.get('ff_r_squared', np.nan), result.get('smb_premium', np.nan),
                    result.get('hml_premium', np.nan), result['market_premium'], result['analysis_date'],
                    result['ticker'], result['date']
                ))
        else:
            # Insert new entry
            if result['model'].lower() == 'capm':
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_analysis_capm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    beta REAL,
                    alpha REAL,
                    expected_return REAL,
                    sharpe_ratio REAL,
                    r_squared REAL,
                    analysis_date TEXT,
                    market_premium REAL,
                    risk_free_rate REAL
                )
                """)
                
                cursor.execute("""
                INSERT INTO fundamental_analysis_capm (
                    ticker, date, beta, alpha, expected_return, 
                    sharpe_ratio, r_squared, analysis_date, market_premium, risk_free_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['ticker'], result['date'], result['beta'],
                    result['alpha'], result['expected_return_capm'],
                    result['sharpe_ratio'], result['r_squared'],
                    result['analysis_date'], result['market_premium'], result['risk_free_rate']
                ))
            else:  # Fama-French model
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_analysis_ff (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    beta REAL,
                    alpha REAL,
                    smb_beta REAL,
                    hml_beta REAL,
                    ff_alpha REAL,
                    expected_return_capm REAL,
                    expected_return_ff REAL,
                    sharpe_ratio REAL,
                    r_squared REAL,
                    ff_r_squared REAL,
                    smb_premium REAL,
                    hml_premium REAL,
                    market_premium REAL,
                    analysis_date TEXT
                )
                """)
                
                cursor.execute("""
                INSERT INTO fundamental_analysis_ff (
                    ticker, date, beta, alpha, smb_beta, hml_beta, ff_alpha,
                    expected_return_capm, expected_return_ff, sharpe_ratio, 
                    r_squared, ff_r_squared, smb_premium, hml_premium, market_premium, analysis_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['ticker'], result['date'], result['beta'],
                    result['alpha'], result.get('smb_beta', np.nan),
                    result.get('hml_beta', np.nan), result.get('ff_alpha', np.nan),
                    result['expected_return_capm'], result.get('expected_return_ff', np.nan),
                    result['sharpe_ratio'], result['r_squared'],
                    result.get('ff_r_squared', np.nan), result.get('smb_premium', np.nan),
                    result.get('hml_premium', np.nan), result['market_premium'], result['analysis_date']
                ))
        
        conn.commit()
        logging.info(f"Stored fundamental analysis results for {result['ticker']}")
        return True
    except Exception as e:
        logging.error(f"Error storing fundamental analysis results: {e}")
        return False

def main():
    """Run the fundamental analysis."""
    # Connect to the database
    db_path = os.path.join('database', 'data.db')
    conn = sqlite3.connect(db_path)
    
    # Get all tickers or use a specific one for testing
    process_all_tickers = False  # Set to False for testing with one ticker
    
    if process_all_tickers:
        # Get all tickers from the database
        tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM nasdaq_100_daily_prices ORDER BY ticker", conn)
        tickers = tickers_df['ticker'].tolist()
    else:
        # Just test with AAPL
        tickers = ['AAPL']
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2020-01-01'  # Use a fixed start date for consistent analysis
    
    # Choose model
    model = 'fama_french'  # 'capm' or 'fama_french'
    
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
    
    if process_all_tickers:
        logging.info(f"Analysis completed. Success: {success_count}, Errors: {error_count}")
    else:
        logging.info("Test analysis completed successfully")

if __name__ == "__main__":
    main()
