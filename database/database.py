# database.py
import psycopg2
import os
from dotenv import load_dotenv

def get_db():
    """Get database connection"""
    try:
        load_dotenv()
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise