import sqlite3
import os
import logging

logging.basicConfig(filename='logs/project.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_connection(db_file: str):
    try:
        conn = sqlite3.connect(db_file)
        logging.info("Connected to database.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

def setup_database(conn, schema_file: str):
    try:
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        logging.info("Database schema created/updated successfully.")
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        raise

if __name__ == "__main__":
    db_path = os.path.join("database", "data.db")
    schema_path = os.path.join("database", "schema.sql")
    conn = create_connection(db_path)
    setup_database(conn, schema_path)
    conn.close()
