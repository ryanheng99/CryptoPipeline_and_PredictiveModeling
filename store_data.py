import psycopg2
from sqlalchemy import create_engine

def store_data(df, db_url):
    engine = create_engine(db_url)
    df.to_sql("crypto_prices", engine, if_exists="append", index=False)
