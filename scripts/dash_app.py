# dash_app.py

import os
import sqlite3
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import logging
from datetime import datetime

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
logging.basicConfig(
    filename='logs/dash_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------------------
# Import functions from our other scripts
# ---------------------------------------------------------------------
from technical_signals import update_technical_signals, apply_trading_strategy
from backtest_technicals import backtest_trading_strategy

# ---------------------------------------------------------------------
# Constants and Database Path
# ---------------------------------------------------------------------
DB_PATH = os.path.join("database", "data.db")
START_DATE = '2000-01-01'
INIT_VALUE = 100000

# ---------------------------------------------------------------------
# Update Technical Signals at Startup (if needed)
# ---------------------------------------------------------------------
# Uncomment if you wish to update signals on app start:
# update_technical_signals(DB_PATH)

# ---------------------------------------------------------------------
# Run Backtest to Get Results
# ---------------------------------------------------------------------
backtest_results_df = backtest_trading_strategy(
    db_path=DB_PATH,
    starting_date=START_DATE,
    investment_value=INIT_VALUE
)
if backtest_results_df is None:
    backtest_results_df = pd.DataFrame()

# ---------------------------------------------------------------------
# Load Price Data and Signals Directly from the Database
# ---------------------------------------------------------------------
try:
    conn = sqlite3.connect(DB_PATH)
    price_df = pd.read_sql_query(
        "SELECT date, ticker, close FROM nasdaq_100_daily_prices",
        conn, parse_dates=['date']
    )
    # Pivot to wide format (date index, tickers as columns)
    price_pivot = price_df.pivot(index='date', columns='ticker', values='close').sort_index()

    # NOTE: Here we alias signal_date as date to match the pivot logic
    signals_df = pd.read_sql_query(
        "SELECT signal_date AS date, ticker, signal FROM technical_signals",
        conn, parse_dates=['date']
    )
    conn.close()

    signals_pivot = signals_df.pivot(index='date', columns='ticker', values='signal').sort_index().fillna(0)
except Exception as e:
    logging.error(f"Error loading price or signals data: {e}")
    price_pivot = pd.DataFrame()
    signals_pivot = pd.DataFrame()

# Create ticker list for dropdown based on price data
tickers = list(price_pivot.columns) if not price_pivot.empty else []

# ---------------------------------------------------------------------
# Define Dash App Layout
# ---------------------------------------------------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Trading Strategy Dashboard"),
    dcc.Tabs(id='tabs', value='tab-dashboard', children=[
        dcc.Tab(label='Dashboard', value='tab-dashboard', children=[
            html.Br(),
            # Backtest Results Table
            dash_table.DataTable(
                id='backtest-table',
                columns=[{"name": col, "id": col} for col in backtest_results_df.columns],
                data=backtest_results_df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'}
            ),
            html.Br(),
            # Interactive Chart for Selected Ticker
            html.Div([
                html.Label("Select Ticker:"),
                dcc.Dropdown(
                    id='ticker-dropdown',
                    options=[{'label': t, 'value': t} for t in tickers],
                    value=tickers[0] if tickers else None,
                    clearable=False
                )
            ], style={'width': '25%', 'display': 'inline-block'}),
            dcc.Graph(id='strategy-graph')
        ]),
        dcc.Tab(label='Raw Data Output', value='tab-raw', children=[
            html.Br(),
            dash_table.DataTable(
                id='raw-data-table',
                columns=[{"name": col, "id": col} for col in backtest_results_df.columns],
                data=backtest_results_df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'}
            )
        ]),
        dcc.Tab(label='Optimization (Placeholder)', value='tab-opt', children=[
            html.Br(),
            html.Div("Optimization results and future efficient frontier analysis will be added here.", style={'textAlign': 'center'})
        ])
    ]),
    html.Div(id='additional-output')
])

# ---------------------------------------------------------------------
# Define Callbacks
# ---------------------------------------------------------------------
@app.callback(
    Output('strategy-graph', 'figure'),
    Input('ticker-dropdown', 'value')
)
def update_graph(selected_ticker):
    if not selected_ticker or price_pivot.empty or signals_pivot.empty:
        return go.Figure()

    # Get price series and corresponding signals
    price_series = price_pivot[selected_ticker].dropna()
    signal_series = signals_pivot[selected_ticker].reindex(price_series.index).fillna(0)

    # Build the figure
    fig = go.Figure()
    # Plot price line
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name=f"{selected_ticker} Price"
    ))
    # Plot buy signals
    buy_mask = (signal_series == 1)
    fig.add_trace(go.Scatter(
        x=price_series.index[buy_mask],
        y=price_series[buy_mask],
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='green',
        marker_size=10,
        name="Buy Signal"
    ))
    # Plot sell signals
    sell_mask = (signal_series == -1)
    fig.add_trace(go.Scatter(
        x=price_series.index[sell_mask],
        y=price_series[sell_mask],
        mode='markers',
        marker_symbol='triangle-down',
        marker_color='red',
        marker_size=10,
        name="Sell Signal"
    ))
    # Add SPY benchmark (if available)
    try:
        conn = sqlite3.connect(DB_PATH)
        spy_df = pd.read_sql_query(
            "SELECT date, close FROM nasdaq_100_daily_prices WHERE ticker = 'SPY' AND date >= ? ORDER BY date",
            conn, params=(START_DATE,), parse_dates=['date']
        )
        conn.close()
        if not spy_df.empty:
            spy_series = spy_df.set_index('date')['close'].sort_index()
            fig.add_trace(go.Scatter(
                x=spy_series.index,
                y=spy_series.values,
                mode='lines',
                name="SPY Benchmark",
                line=dict(color='orange')
            ))
    except Exception as e:
        logging.error(f"Error loading SPY benchmark data: {e}")

    fig.update_layout(
        title=f"Interactive Trading Strategy for {selected_ticker}",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

# ---------------------------------------------------------------------
# Run the Dash App
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
