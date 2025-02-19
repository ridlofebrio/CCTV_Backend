import psycopg2  # type: ignore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create connection
conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)
cursor = conn.cursor()

try:
    # Drop existing tables and types in correct order
    cursor.execute("""
    DO $$ 
    BEGIN
        -- Drop tables if they exist
        DROP TABLE IF EXISTS detection CASCADE;
        DROP TABLE IF EXISTS ppa CASCADE;
        DROP TABLE IF EXISTS cctv CASCADE;
        
        -- Drop enum type if exists
        DROP TYPE IF EXISTS ppa_label CASCADE;
    END
    $$;
    """)

    # Create cctv table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cctv (
        id SERIAL PRIMARY KEY,
        merek VARCHAR(100),
        gedung VARCHAR(100),
        lantai INT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        gambar VARCHAR(255)
    );
    """)

    # Create ppa table with ENUM type
    cursor.execute("""
    DO $$ 
    BEGIN
        CREATE TYPE ppa_label AS ENUM ('Tidak Memakai Helm', 'Tidak Memakai Masker', 'Tidak Memakai Rompi');
    END
    $$;
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ppa (
        id SERIAL PRIMARY KEY,
        label ppa_label
    );
    """)

    # Create detection table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detection (
        id SERIAL PRIMARY KEY,
        id_cctv INT REFERENCES cctv(id) ON DELETE CASCADE,
        id_ppa INT REFERENCES ppa(id) ON DELETE CASCADE,
        deteksi_jatuh BOOLEAN,
        deteksi_overtime INT,
        link_playback VARCHAR(255),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidan DOUBLE PRECISION
    );
    """)

    conn.commit()
    print("Tables dropped and recreated successfully.")

except psycopg2.Error as e:
    print(f"Database error: {e}")
    conn.rollback()

finally:
    # Close database connection
    cursor.close()
    conn.close()
