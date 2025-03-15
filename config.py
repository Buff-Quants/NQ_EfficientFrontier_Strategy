## config.py

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
# Simple Moving Averages
SMA_SHORT_WINDOW = 5
SMA_LONG_WINDOW = 20

# Relative Strength Index (RSI)
RSI_WINDOW = 14

# Bollinger Bands
BB_WINDOW = 20
BB_MULTIPLIER = 2

# MACD
MACD_SHORT_SPAN = 12
MACD_LONG_SPAN = 26
MACD_SIGNAL_SPAN = 9

# On-Balance Volume (OBV)
OBV_MA_WINDOW = 20

# Stochastic Oscillator
STOCH_WINDOW = 14
STOCH_SMOOTH_WINDOW = 3

# Average Directional Index (ADX)
ADX_WINDOW = 14

# Overall Signal Threshold for combined technical indicators
TECH_SIGNAL_THRESHOLD = 3

#########################
# Fundamental Analysis Parameters
#########################
# CAPM parameters
RISK_FREE_RATE = 0.02       # risk-free rate (e.g., 2%)
MARKET_RISK_PREMIUM = 0.06    # market risk premium (e.g., 6%)

# Additional parameters for Fama-French (if needed, adjust as appropriate)
# (You might include default factor multipliers or threshold values here if you use them)

#########################
# Optimization & Monte Carlo Settings
#########################
NUM_SIMULATIONS = 10000     # number of Monte Carlo simulation iterations

# SMA optimization parameter ranges (as tuples or lists)
SMA_SHORT_RANGE = list(range(3, 15))      # example range from 3 to 14
SMA_LONG_RANGE = list(range(15, 50, 5))     # example range from 15 to 45, in steps of 5

#########################
# Logging & Debugging
#########################
LOG_FILE = f"{LOG_DIR}/project.log"
LOG_LEVEL = 'INFO'  # or 'DEBUG' if verbose output is needed
DEBUG_MODE = False

#########################
# Additional Configurations
#########################
# If you add more modules (e.g., optimization, Monte Carlo) you can include further settings here.
# For instance, parameters for portfolio rebalancing, risk constraints, etc.
