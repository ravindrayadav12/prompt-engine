import psycopg2
from psycopg2 import pool
connection_pool = pool.SimpleConnectionPool(
    1, 10,
    database="promt_engine",
    user="postgres",
    password="gonish@21",
    host="localhost",
    port="5432"
)

def get_conn():
    return connection_pool.getconn()

def release_conn(conn):
    connection_pool.putconn(conn)
