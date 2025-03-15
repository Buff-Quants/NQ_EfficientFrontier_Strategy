import os
import sqlite3
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import logging
from datetime import datetime

# Import your technical signal function
from technical_signals import apply_trading_strategy
# Import your backtest function (just for loading backtest_results_df)
from backtest_technicals import backtest_trading_strategy

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
logging.basicConfig(
    filename='logs/dash_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------------------
# Constants and Database Path
# ---------------------------------------------------------------------
DB_PATH = os.path.join("database", "data.db")
START_DATE = '2000-01-01'
INIT_VALUE = 100000

# ---------------------------------------------------------------------
# Run Backtest to Get Results (for the table)
# ---------------------------------------------------------------------
backtest_results_df = backtest_trading_strategy(
    db_path=DB_PATH,
    starting_date=START_DATE,
    investment_value=INIT_VALUE
)
if backtest_results_df is None:
    backtest_results_df = pd.DataFrame()

# ---------------------------------------------------------------------
# Fetch all tickers from the database for dropdown
# ---------------------------------------------------------------------
try:
    conn = sqlite3.connect(DB_PATH)
    ticker_query = "SELECT DISTINCT ticker FROM nasdaq_100_daily_prices"
    df_tickers = pd.read_sql_query(ticker_query, conn)
    conn.close()
    tickers = sorted(df_tickers['ticker'].unique().tolist())
except Exception as e:
    logging.error(f"Error loading ticker list: {e}")
    tickers = []

# ---------------------------------------------------------------------
# Define Dash App
# ---------------------------------------------------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Technical Strategy Dashboard"),
    
    dcc.Tabs(id='tabs', value='tab-technical', children=[
        dcc.Tab(label='Technical Strategy', value='tab-technical', children=[
            html.Br(),

            # Sorting Options for the Backtest Table
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
            
            # Main Graph: Ticker + Benchmarks + Signals
            dcc.Graph(id='strategy-graph'),
        ])
    ]),
    
    html.Div(id='additional-output')
])

# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

# Sort the backtest table
@app.callback(
    Output('backtest-table', 'data'),
    [Input('sort-dropdown', 'value'),
     Input('sort-order', 'value')]
)
def sort_backtest_table(sort_col, sort_order):
    if backtest_results_df.empty or sort_col not in backtest_results_df.columns:
        return backtest_results_df.to_dict('records')
    sorted_df = backtest_results_df.sort_values(by=sort_col, ascending=(sort_order=='asc'))
    return sorted_df.to_dict('records')


@app.callback(
    Output('strategy-graph', 'figure'),
    Input('ticker-dropdown', 'value')
)
def update_graph(selected_ticker):
    """
    1. Load the chosen ticker's data from the database (including volume).
    2. Apply your custom technical strategy to get 'overall_signal'.
    3. Convert 'overall_signal' to boolean buy/sell signals.
    4. Plot the % change of the ticker vs. SPY and ^NDX plus buy/sell markers.
    """
    fig = go.Figure()

    if not selected_ticker:
        return fig

    # --------------------------------------------------
    # 1) Load Ticker Data from DB
    # --------------------------------------------------
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"""
            SELECT date, close, volume
            FROM nasdaq_100_daily_prices
            WHERE ticker = '{selected_ticker}'
            ORDER BY date
        """
        df_ticker = pd.read_sql_query(query, conn, parse_dates=['date'])
        conn.close()
    except Exception as e:
        logging.error(f"Error loading data for {selected_ticker}: {e}")
        return fig
    
    if df_ticker.empty:
        return fig

    # --------------------------------------------------
    # 2) Apply Trading Strategy
    # --------------------------------------------------
    df_ticker = df_ticker.sort_values(by='date').reset_index(drop=True)
    df_ticker = apply_trading_strategy(df_ticker)  # from technical_signals.py

    # --------------------------------------------------
    # 3) Convert overall_signal to buy/sell booleans
    #    overall_signal == 1 => Buy
    #    overall_signal == -1 => Sell
    # --------------------------------------------------
    df_ticker['buy_signal'] = (
        (df_ticker['overall_signal'] == 1) &
        (df_ticker['overall_signal'].shift(1) != 1)
    )
    df_ticker['sell_signal'] = (
        (df_ticker['overall_signal'] == -1) &
        (df_ticker['overall_signal'].shift(1) != -1)
    )

    # --------------------------------------------------
    # 4) Compute % change for the Ticker
    # --------------------------------------------------
    df_ticker['pct_change'] = (df_ticker['close'] / df_ticker['close'].iloc[0] - 1) * 100

    # --------------------------------------------------
    # 5) Load Benchmarks (SPY and ^NDX) for the same dates
    # --------------------------------------------------
    try:
        conn = sqlite3.connect(DB_PATH)
        bench_df = pd.read_sql_query(
            """
            SELECT ticker, date, close
            FROM nasdaq_100_daily_prices
            WHERE ticker IN ('SPY', '^NDX')
            ORDER BY date
            """,
            conn, parse_dates=['date']
        )
        conn.close()
    except Exception as e:
        logging.error(f"Error loading benchmark data: {e}")
        bench_df = pd.DataFrame()

    # If we have benchmark data, compute % change from each series start
    if not bench_df.empty:
        # SPY
        spy_df = bench_df[bench_df['ticker'] == 'SPY'].copy()
        spy_df = spy_df.sort_values(by='date').reset_index(drop=True)
        if not spy_df.empty:
            spy_df['pct_change'] = (spy_df['close'] / spy_df['close'].iloc[0] - 1) * 100

        # ^NDX
        ndx_df = bench_df[bench_df['ticker'] == '^NDX'].copy()
        ndx_df = ndx_df.sort_values(by='date').reset_index(drop=True)
        if not ndx_df.empty:
            ndx_df['pct_change'] = (ndx_df['close'] / ndx_df['close'].iloc[0] - 1) * 100
    else:
        spy_df = pd.DataFrame()
        ndx_df = pd.DataFrame()

    # --------------------------------------------------
    # 6) Build the Plotly Figure
    # --------------------------------------------------
    # Plot Ticker's % change
    fig.add_trace(go.Scatter(
        x=df_ticker['date'],
        y=df_ticker['pct_change'],
        mode='lines',
        name=f"{selected_ticker} % Change"
    ))

    # Plot SPY
    if not spy_df.empty:
        fig.add_trace(go.Scatter(
            x=spy_df['date'],
            y=spy_df['pct_change'],
            mode='lines',
            name="SPY Benchmark",
            line=dict(color='orange', dash='dash')
        ))

    # Plot ^NDX
    if not ndx_df.empty:
        fig.add_trace(go.Scatter(
            x=ndx_df['date'],
            y=ndx_df['pct_change'],
            mode='lines',
            name="NASDAQ Benchmark",
            line=dict(color='purple', dash='dot')
        ))

    # Overlay Buy Signals
    buys = df_ticker[df_ticker['buy_signal'] == True]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['date'],
            y=buys['pct_change'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))

    # Overlay Sell Signals
    sells = df_ticker[df_ticker['sell_signal'] == True]
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['date'],
            y=sells['pct_change'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))

    fig.update_layout(
        title=f"Performance Comparison for {selected_ticker} with Trading Signals",
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

# ---------------------------------------------------------------------
# Run the Dash App
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
