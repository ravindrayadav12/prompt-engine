import os
import psycopg2
from psycopg2 import pool

DATABASE_URL = os.getenv("DATABASE_URL")

connection_pool = pool.SimpleConnectionPool(
    1, 10,
    dsn=DATABASE_URL
)

def get_conn():
    return connection_pool.getconn()

def release_conn(conn):
    connection_pool.putconn(conn)
