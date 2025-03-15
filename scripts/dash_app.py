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
# Import functions from other scripts
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
# Load Price Data and Signals from Database
# ---------------------------------------------------------------------
try:
    conn = sqlite3.connect(DB_PATH)
    price_df = pd.read_sql_query(
        "SELECT date, ticker, close, volume FROM nasdaq_100_daily_prices",
        conn, parse_dates=['date']
    )
    conn.close()

    # Pivot close data
    price_pivot = price_df.pivot(index='date', columns='ticker', values='close').sort_index()
except Exception as e:
    logging.error(f"Error loading price data: {e}")
    price_pivot = pd.DataFrame()

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

            # Sorting Options for Data Table
            html.Div([
                html.Label("Sort By:"),
                dcc.Dropdown(
                    id='sort-dropdown',
                    options=[{'label': col, 'value': col} for col in backtest_results_df.columns],
                    value='Profit Percentage',
                    clearable=False
                ),
                dcc.RadioItems(
                    id='sort-order',
                    options=[
                        {'label': 'Ascending', 'value': 'asc'},
                        {'label': 'Descending', 'value': 'desc'}
                    ],
                    value='desc',
                    inline=True
                ),
            ], style={'width': '50%', 'margin-bottom': '20px'}),

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

            # Ticker Selection
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
    if not selected_ticker or price_pivot.empty:
        return go.Figure()

    # Extract close price for selected ticker
    close_series = price_pivot[selected_ticker].dropna()
    close_pct_change = (close_series / close_series.iloc[0] - 1) * 100  # Convert to %

    # Fetch SPY & NASDAQ Benchmarks
    try:
        conn = sqlite3.connect(DB_PATH)
        benchmarks_df = pd.read_sql_query(
            "SELECT date, ticker, close FROM nasdaq_100_daily_prices WHERE ticker IN ('SPY', '^NDX')",
            conn, parse_dates=['date']
        )
        conn.close()

        if not benchmarks_df.empty:
            spy_series = benchmarks_df[benchmarks_df['ticker'] == 'SPY'].set_index('date')['close']
            ndx_series = benchmarks_df[benchmarks_df['ticker'] == '^NDX'].set_index('date')['close']

            spy_pct_change = (spy_series / spy_series.iloc[0] - 1) * 100
            ndx_pct_change = (ndx_series / ndx_series.iloc[0] - 1) * 100
        else:
            spy_pct_change = pd.Series(dtype=float)
            ndx_pct_change = pd.Series(dtype=float)

    except Exception as e:
        logging.error(f"Error loading SPY & NASDAQ data: {e}")
        spy_pct_change = pd.Series(dtype=float)
        ndx_pct_change = pd.Series(dtype=float)

    # Build Plotly Figure
    fig = go.Figure()

    # Plot Ticker as % Change
    fig.add_trace(go.Scatter(
        x=close_series.index,
        y=close_pct_change,
        mode='lines',
        name=f"{selected_ticker} % Change"
    ))

    # Plot Benchmarks
    if not spy_pct_change.empty:
        fig.add_trace(go.Scatter(
            x=spy_pct_change.index,
            y=spy_pct_change,
            mode='lines',
            name="SPY Benchmark",
            line=dict(color='orange', dash='dash')
        ))

    if not ndx_pct_change.empty:
        fig.add_trace(go.Scatter(
            x=ndx_pct_change.index,
            y=ndx_pct_change,
            mode='lines',
            name="NASDAQ Benchmark",
            line=dict(color='purple', dash='dot')
        ))

    fig.update_layout(
        title=f"Performance Comparison for {selected_ticker}",
        yaxis_title="Percentage Change (%)",
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


@app.callback(
    Output('backtest-table', 'data'),
    [Input('sort-dropdown', 'value'), Input('sort-order', 'value')]
)
def update_backtest_table(sort_by, order):
    if backtest_results_df.empty:
        return []

    sorted_df = backtest_results_df.sort_values(by=sort_by, ascending=(order == 'asc'))
    return sorted_df.to_dict('records')


# ---------------------------------------------------------------------
# Run the Dash App
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
