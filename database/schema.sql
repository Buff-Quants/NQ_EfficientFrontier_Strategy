-- Table: backtesting_results
CREATE TABLE IF NOT EXISTS backtesting_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_date TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL
);

-- Table: fundamental_analysis_capm
CREATE TABLE IF NOT EXISTS fundamental_analysis_capm (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    expected_return REAL NOT NULL,
    calculation_date TEXT NOT NULL
);

-- Table: fundamental_analysis_ff
CREATE TABLE IF NOT EXISTS fundamental_analysis_ff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    expected_return_ff REAL NOT NULL,
    calculation_date TEXT NOT NULL
);

-- Table: fundamental_results
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
    UNIQUE(ticker, analysis_date)
);

-- Table: nasdaq_100_daily_prices
CREATE TABLE IF NOT EXISTS nasdaq_100_daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    volume INTEGER,
    UNIQUE(ticker, date)
);

-- Table: nasdaq_100_tickers
CREATE TABLE IF NOT EXISTS nasdaq_100_tickers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    UNIQUE(date, ticker)
);

-- Table: portfolio_optimization
CREATE TABLE IF NOT EXISTS portfolio_optimization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_date TEXT NOT NULL,
    portfolio_weights TEXT,
    expected_return REAL,
    volatility REAL,
    sharpe_ratio REAL
);

-- Table: technical_signals
CREATE TABLE IF NOT EXISTS technical_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    sma REAL,
    rsi REAL,
    macd REAL,
    atr REAL,
    signal TEXT,
    UNIQUE(ticker, signal_date)
);

-- Optional: Create Indexes for faster queries

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date 
    ON nasdaq_100_daily_prices (ticker, date);

CREATE INDEX IF NOT EXISTS idx_tickers_date 
    ON nasdaq_100_tickers (date, ticker);

CREATE INDEX IF NOT EXISTS idx_capm_ticker_date 
    ON fundamental_analysis_capm (ticker, calculation_date);

CREATE INDEX IF NOT EXISTS idx_ff_ticker_date 
    ON fundamental_analysis_ff (ticker, calculation_date);

CREATE INDEX IF NOT EXISTS idx_signals_ticker_date 
    ON technical_signals (ticker, signal_date);
