-- Table: Nasdaq-100 Tickers
CREATE TABLE IF NOT EXISTS nasdaq_100_tickers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    UNIQUE(date, ticker)
);

-- Table: Daily Price Data
CREATE TABLE IF NOT EXISTS nasdaq_100_daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    volume INTEGER,
    UNIQUE(ticker, date)
);

-- Table: Fundamental Analysis Results (CAPM, Fama-French)
CREATE TABLE IF NOT EXISTS fundamental_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    beta REAL,
    expected_return REAL,
    ff_factor1 REAL,  -- e.g., Size
    ff_factor2 REAL,  -- e.g., Value
    ff_factor3 REAL,  -- e.g., Profitability
    ff_factor4 REAL,  -- e.g., Investment
    ff_factor5 REAL,  -- e.g., (if applicable)
    UNIQUE(ticker, analysis_date)
);

-- Table: Monte Carlo & Portfolio Optimization Results
CREATE TABLE IF NOT EXISTS portfolio_optimization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_date TEXT NOT NULL,
    portfolio_weights TEXT,  -- Store as JSON mapping tickers to weights
    expected_return REAL,
    volatility REAL,
    sharpe_ratio REAL
);

-- Table: Technical Indicator Signals
CREATE TABLE IF NOT EXISTS technical_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    sma REAL,         -- Simple Moving Average
    rsi REAL,         -- Relative Strength Index
    macd REAL,        -- MACD value
    atr REAL,         -- Average True Range
    signal TEXT,      -- e.g., "buy", "sell", or "hold"
    UNIQUE(ticker, signal_date)
);

-- Table: Backtesting Results
CREATE TABLE IF NOT EXISTS backtesting_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_date TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL
);

-- Indexes to speed up queries
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON nasdaq_100_daily_prices (ticker, date);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date ON fundamental_results (ticker, analysis_date);
CREATE INDEX IF NOT EXISTS idx_tech_signals ON technical_signals (ticker, signal_date);
