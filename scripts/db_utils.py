import sqlite3
import os
import logging
from typing import List, Tuple
from config import DB_PATH, SCHEMA_PATH, LOG_DIR, LOG_FILE, LOG_LEVEL

# Configure logging using centralized config values.
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_connection(db_file: str) -> sqlite3.Connection:
    """
    Create and return a database connection.
    """
    try:
        conn = sqlite3.connect(db_file)
        logging.info("Connected to database.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

def setup_database(conn: sqlite3.Connection, schema_file: str) -> None:
    """
    Set up or update the database schema using the SQL script provided in schema_file.
    
    Future enhancements:
      - Consider incremental updates if the full schema script becomes too heavy.
      - Review indexing and vacuum operations as data grows.
    """
    try:
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        logging.info("Database schema created/updated successfully.")
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        raise

def execute_query(conn: sqlite3.Connection, query: str, params: Tuple = ()) -> None:
    """
    Execute a single query with optional parameters.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        logging.info(f"Executed query: {query}")
    except sqlite3.Error as e:
        logging.error(f"Error executing query: {e}")
        raise

def create_table(conn: sqlite3.Connection, table_name: str, columns: List[Tuple[str, str]]) -> None:
    """
    Create a table dynamically with specified columns.
    Each element in columns is a tuple: (column_name, column_type).
    """
    columns_str = ", ".join([f"{name} {col_type}" for name, col_type in columns])
    query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {columns_str})"
    execute_query(conn, query)
    logging.info(f"Created table {table_name}")

def insert_data(conn: sqlite3.Connection, table_name: str, columns: List[str], data: List[Tuple]) -> None:
    """
    Insert multiple rows of data into a table.
    
    :param table_name: Name of the table to insert data into.
    :param columns: List of column names to insert data.
    :param data: List of tuples, each tuple is one row of data.
    """
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

if __name__ == "__main__":
    # This block will run when the script is executed directly.
    # It performs a database setup using the schema provided in the config.
    conn = create_connection(DB_PATH)
    setup_database(conn, SCHEMA_PATH)
    conn.close()
