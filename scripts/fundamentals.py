#!/usr/bin/env python3
"""
Fundamental analysis module for quantitative investment strategy.
Implements CAPM and Fama-French models for expected returns estimation.
Results are stored in the 'fundamental_results' table in the SQL database.
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
from typing import Dict, Optional
# Import configuration constants
from config import DB_PATH, LOG_DIR, LOG_LEVEL, FUNDAMENTALS_START_DATE, FUNDAMENTALS_MODEL

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)
# Configure logging to a dedicated fundamentals log file
fund_log_file = os.path.join(LOG_DIR, 'fundamentals.log')
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(fund_log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Cache for downloaded data
_CACHE = {
    'market_data': {},
    'risk_free_data': {},
    'ff_factors': {}
}

def get_price_data(conn: sqlite3.Connection, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data for a ticker from the database using parameterized queries."""
    try:
        query = """
        SELECT date, close, volume
        FROM nasdaq_100_daily_prices
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """
        logging.info(f"Executing SQL for price data for {ticker}")
        df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        logging.info(f"Retrieved {len(df)} rows of price data for {ticker}")
        if df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates('date', keep='last')
        df.set_index('date', inplace=True)
        df.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error getting price data for {ticker}: {e}")
        return pd.DataFrame()

def get_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Get market data (S&P 500 proxy) with caching."""
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
    """Get risk-free rate data with caching."""
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _CACHE['risk_free_data']:
        return _CACHE['risk_free_data'][cache_key]
    try:
        adjusted_end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        logging.info("Downloading risk-free rate data")
        rf_data = yf.download('^IRX', start=start_date, end=adjusted_end_date)
        if rf_data.empty:
            return pd.DataFrame()
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
        logging.info("Downloading Fama-French factor proxies")
        iwm = yf.download('IWM', start=start_date, end=end_date)['Close']
        spy = yf.download('SPY', start=start_date, end=end_date)['Close']
        iwd = yf.download('IWD', start=start_date, end=end_date)['Close']
        iwf = yf.download('IWF', start=start_date, end=end_date)['Close']
        iwm_ret = iwm.pct_change().dropna()
        spy_ret = spy.pct_change().dropna()
        iwd_ret = iwd.pct_change().dropna()
        iwf_ret = iwf.pct_change().dropna()
        if isinstance(iwm_ret, pd.DataFrame): iwm_ret = iwm_ret.iloc[:, 0]
        if isinstance(spy_ret, pd.DataFrame): spy_ret = spy_ret.iloc[:, 0]
        if isinstance(iwd_ret, pd.DataFrame): iwd_ret = iwd_ret.iloc[:, 0]
        if isinstance(iwf_ret, pd.DataFrame): iwf_ret = iwf_ret.iloc[:, 0]
        smb_proxy = iwm_ret - spy_ret
        hml_proxy = iwd_ret - iwf_ret
        rf_data = get_risk_free_rate(start_date, end_date)
        factors = pd.DataFrame({'SMB': smb_proxy, 'HML': hml_proxy})
        if 'daily_rate' in rf_data.columns:
            rf_series = rf_data['daily_rate']
            factors = factors.join(rf_series, how='left', rsuffix='_rf')
            factors.rename(columns={'daily_rate': 'RF'}, inplace=True)
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
    stock_returns = stock_returns.dropna()
    market_returns = market_returns.dropna()
    risk_free_rate = risk_free_rate.dropna()
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
        data = pd.DataFrame({'excess_return': stock_excess_returns})
        data = data.join(factors_data).dropna()
        X_cols = ['SMB', 'HML']
        if 'MKT-RF' in data.columns:
            X_cols = ['MKT-RF'] + X_cols
        X = data[X_cols]
        y = data['excess_return']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        coefficients = {col: model.params[col] for col in X_cols}
        result = {
            'alpha': model.params['const'],
            'r_squared': model.rsquared
        }
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
        market_beta = factor_loadings.get('market_beta', 0.0)
        smb_beta = factor_loadings.get('smb_beta', 0.0)
        hml_beta = factor_loadings.get('hml_beta', 0.0)
        market_premium = factor_premiums.get('MKT-RF', 0.0)
        smb_premium = factor_premiums.get('SMB', 0.0)
        hml_premium = factor_premiums.get('HML', 0.0)
        return risk_free_rate + market_beta * market_premium + smb_beta * smb_premium + hml_beta * hml_premium
    except Exception as e:
        logging.error(f"Error calculating FF expected returns: {e}")
        return np.nan

def calculate_sharpe_ratio(excess_returns):
    """Calculate Sharpe ratio (annualized)."""
    try:
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {e}")
        return np.nan

def run_fundamental_analysis(conn: sqlite3.Connection, ticker: str, start_date: str, end_date: str, model: str = FUNDAMENTALS_MODEL) -> Optional[Dict]:
    """Run fundamental analysis for a ticker."""
    try:
        logging.info(f"Running {model.upper()} analysis for {ticker} from {start_date} to {end_date}")
        stock_data = get_price_data(conn, ticker, start_date, end_date)
        if stock_data.empty:
            logging.warning(f"No price data available for {ticker}")
            return None
        market_data = get_market_data(start_date, end_date)
        if market_data.empty:
            logging.warning("No market data available for analysis period")
            return None
        risk_free_data = get_risk_free_rate(start_date, end_date)
        if risk_free_data.empty or 'daily_rate' not in risk_free_data.columns:
            logging.warning("No risk-free rate data available for analysis period")
            return None
        
        stock_returns = calculate_returns(stock_data)
        market_returns = calculate_returns(market_data)
        risk_free_rate = risk_free_data['daily_rate']
        
        stock_returns, market_returns, risk_free_rate = validate_data(
            stock_returns, market_returns, risk_free_rate
        )
        
        if len(stock_returns) < 30:
            logging.warning(f"Insufficient data for {ticker}: only {len(stock_returns)} days")
            return None
        
        stock_excess_returns = stock_returns - risk_free_rate
        market_excess_returns = market_returns - risk_free_rate
        avg_rf_rate = risk_free_rate.mean()
        market_premium = market_excess_returns.mean()
        
        # Prepare result dictionary for storage in fundamental_results table.
        result = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'beta': None,
            'expected_return': None,
            'ff_factor1': None,
            'ff_factor2': None,
            'ff_factor3': None,
            'ff_factor4': None,
            'ff_factor5': None,
            'model': model
        }
        
        # Run CAPM analysis
        beta, alpha, r_squared = calculate_beta_capm(stock_excess_returns, market_excess_returns)
        expected_return_capm = calculate_expected_returns_capm(beta, avg_rf_rate, market_premium)
        
        if model.lower() == 'capm':
            result['beta'] = beta
            result['expected_return'] = expected_return_capm * 252  # Annualized
            result['ff_factor1'] = alpha * 252  # Annualized alpha
            result['ff_factor2'] = r_squared
            result['ff_factor3'] = market_premium * 252  # Annualized market premium
            result['ff_factor4'] = avg_rf_rate * 252       # Annualized risk-free rate
            result['ff_factor5'] = None
        elif model.lower() == 'fama_french':
            ff_factors = get_fama_french_factors(start_date, end_date)
            if not ff_factors.empty:
                ff_factors['MKT-RF'] = market_excess_returns
                ff_loadings = calculate_fama_french_factors(stock_excess_returns, ff_factors)
                factor_premiums = {
                    'MKT-RF': market_premium * 252,
                    'SMB': ff_factors['SMB'].mean() * 252,
                    'HML': ff_factors['HML'].mean() * 252
                }
                expected_return_ff = calculate_expected_returns_ff(
                    ff_loadings,
                    {k: v/252 for k, v in factor_premiums.items()},
                    avg_rf_rate
                )
                result['beta'] = ff_loadings.get('market_beta', beta)
                result['expected_return'] = expected_return_ff * 252
                result['ff_factor1'] = ff_loadings.get('smb_beta', np.nan)
                result['ff_factor2'] = ff_loadings.get('hml_beta', np.nan)
                result['ff_factor3'] = ff_loadings.get('alpha', np.nan) * 252
                result['ff_factor4'] = ff_loadings.get('r_squared', np.nan)
                result['ff_factor5'] = None
        else:
            logging.error(f"Unknown fundamental model: {model}")
            return None
        
        result['sharpe_ratio'] = calculate_sharpe_ratio(stock_excess_returns)
        return result
    except Exception as e:
        logging.error(f"Error in fundamental analysis for {ticker}: {e}")
        return None

def store_fundamental_results(conn: sqlite3.Connection, result: Dict) -> bool:
    """
    Store fundamental analysis results in the 'fundamental_results' table.
    The schema is assumed to have the following columns:
      - ticker (TEXT, NOT NULL)
      - analysis_date (TEXT, NOT NULL)
      - beta (REAL)
      - expected_return (REAL)
      - ff_factor1 (REAL)
      - ff_factor2 (REAL)
      - ff_factor3 (REAL)
      - ff_factor4 (REAL)
      - ff_factor5 (REAL)
      - model (TEXT)
    with a UNIQUE constraint on (ticker, analysis_date).
    """
    try:
        cursor = conn.cursor()
        # Create table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                beta REAL,
                expected_return REAL,
                ff_factor1 REAL,
                ff_factor2 REAL,
                ff_factor3 REAL,
                ff_factor4 REAL,
                ff_factor5 REAL,
                model TEXT,
                UNIQUE(ticker, analysis_date)
            )
        """)
        # Check if entry already exists
        cursor.execute("""
            SELECT id FROM fundamental_results
            WHERE ticker = ? AND analysis_date = ?
        """, (result['ticker'], result['analysis_date']))
        existing_entry = cursor.fetchone()
        
        if existing_entry:
            cursor.execute("""
                UPDATE fundamental_results
                SET beta = ?, expected_return = ?, ff_factor1 = ?, ff_factor2 = ?,
                    ff_factor3 = ?, ff_factor4 = ?, ff_factor5 = ?, model = ?
                WHERE ticker = ? AND analysis_date = ?
            """, (
                result['beta'], result['expected_return'], result['ff_factor1'],
                result['ff_factor2'], result['ff_factor3'], result['ff_factor4'],
                result['ff_factor5'], result['model'],
                result['ticker'], result['analysis_date']
            ))
        else:
            cursor.execute("""
                INSERT INTO fundamental_results (
                    ticker, analysis_date, beta, expected_return,
                    ff_factor1, ff_factor2, ff_factor3, ff_factor4, ff_factor5, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result['ticker'], result['analysis_date'], result['beta'], result['expected_return'],
                result['ff_factor1'], result['ff_factor2'], result['ff_factor3'], result['ff_factor4'],
                result['ff_factor5'], result['model']
            ))
        conn.commit()
        logging.info(f"Stored fundamental analysis results for {result['ticker']}")
        return True
    except Exception as e:
        logging.error(f"Error storing fundamental analysis results: {e}")
        return False

def main():
    """Run the fundamental analysis for all tickers (or a test subset) and store results in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Process all tickers from nasdaq_100_daily_prices
        tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM nasdaq_100_daily_prices ORDER BY ticker", conn)
        tickers = tickers_df['ticker'].tolist()
        logging.info(f"Found {len(tickers)} tickers to analyze")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = FUNDAMENTALS_START_DATE
        model = FUNDAMENTALS_MODEL
        
        success_count = 0
        error_count = 0
        
        for i, ticker in enumerate(tickers):
            logging.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
            try:
                result = run_fundamental_analysis(conn, ticker, start_date, end_date, model)
                if result:
                    store_fundamental_results(conn, result)
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
    except Exception as e:
        logging.error(f"Main process in fundamentals failed: {e}")

if __name__ == "__main__":
    main()
