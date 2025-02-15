## db_utils.py

import sqlite3
import logging
from typing import List, Tuple

logging.basicConfig(filename='logs/nq_tickers.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_connection(db_file: str):
    """Create and return a database connection."""
    try:
        conn = sqlite3.connect(db_file)
        logging.info("Connected to database.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

def execute_query(conn, query: str, params: Tuple = ()):
    """Execute a single query with optional parameters."""
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        logging.info(f"Executed query: {query}")
    except sqlite3.Error as e:
        logging.error(f"Error executing query: {e}")
        raise

def create_table(conn, table_name: str, columns: List[Tuple[str, str]]):
    """Create a table dynamically with specified columns."""
    columns_str = ", ".join([f"{name} {type}" for name, type in columns])
    query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {columns_str})"
    execute_query(conn, query)
    logging.info(f"Created table {table_name}")

def insert_data(conn, table_name: str, columns: List[str], data: List[Tuple]):
    """Insert multiple rows of data into a table."""
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)
    query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
    try:
        cursor = conn.cursor()
        cursor.executemany(query, data)
        conn.commit()
        logging.info(f"Inserted {len(data)} records into {table_name}")
    except sqlite3.Error as e:
        logging.error(f"Error inserting data into {table_name}: {e}")
        raise
