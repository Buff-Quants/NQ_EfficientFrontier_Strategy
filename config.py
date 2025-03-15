"""
Configuration file for NQ_Efficient_Frontier project.

This file centralizes all constants and configurable parameters used throughout the project,
including database paths, technical indicator parameters, fundamental analysis constants,
optimization settings, and logging configurations.
"""

#########################
# Paths & Database
#########################
DB_PATH = 'database/data.db'
SCHEMA_PATH = 'database/schema.sql'

# Directories for logs and results (if used in code)
LOG_DIR = 'logs'
RESULTS_DIR = 'results'

#########################
# Date & Investment Parameters
#########################
START_DATE = '2000-01-01'
INIT_VALUE = 100000  # initial capital for backtesting/portfolio simulation
DATA_FREQ = '1D'     # frequency for price data (e.g., daily)

#########################
# Technical Indicator Parameters
#########################
SMA_SHORT_WINDOW = 5
SMA_LONG_WINDOW = 20
RSI_WINDOW = 14
BB_WINDOW = 20
BB_MULTIPLIER = 2
MACD_SHORT_SPAN = 12
MACD_LONG_SPAN = 26
MACD_SIGNAL_SPAN = 9
OBV_MA_WINDOW = 20
STOCH_WINDOW = 14
STOCH_SMOOTH_WINDOW = 3
ADX_WINDOW = 14
TECH_SIGNAL_THRESHOLD = 3

#########################
# Fundamental Analysis Parameters
#########################
RISK_FREE_RATE = 0.02       # risk-free rate (e.g., 2%)
MARKET_RISK_PREMIUM = 0.06    # market risk premium (e.g., 6%)

# Default values for running fundamental analysis
FUNDAMENTALS_START_DATE = '2020-01-01'
FUNDAMENTALS_MODEL = 'fama_french'  # or 'capm'

#########################
# Optimization & Monte Carlo Settings
#########################
NUM_SIMULATIONS = 10000
SMA_SHORT_RANGE = list(range(3, 15))
SMA_LONG_RANGE = list(range(15, 50, 5))

#########################
# Logging & Debugging
#########################
LOG_FILE = f"{LOG_DIR}/project.log"
LOG_LEVEL = 'INFO'
DEBUG_MODE = False

#########################
# Additional Configurations
#########################
# For future modules like portfolio rebalancing or risk constraints.
