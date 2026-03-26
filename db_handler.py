"""
db_handler.py  –  PostgreSQL connection helpers for tbco schema
"""
import logging
import pg8000.dbapi as pg

logger = logging.getLogger(__name__)


def initialize_db_connection(db_config):
    """Open and return (conn, cur) for the given db_config dict."""
    try:
        conn = pg.connect(
            host=db_config['host'],
            port=db_config.get('port', 5432),
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            timeout=60
        )
        cur = conn.cursor()
        logger.info("DB connection established.")
        return conn, cur
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        raise


def close_db_connection(conn, cur):
    """Close cursor and connection safely."""
    try:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.info("DB connection closed.")
    except Exception as e:
        if 'closed' not in str(e).lower():
            logger.error(f"Error closing DB connection: {e}")
