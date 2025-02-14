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
    # Create cctv table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cctv (
        id SERIAL PRIMARY KEY,
        merek VARCHAR(100),
        gedung VARCHAR(100),
        lantai INT,
        koordinat POINT
    );
    """)

    # Create ppa table with ENUM type
    cursor.execute("""
    DO $$ 
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ppa_label') THEN
            CREATE TYPE ppa_label AS ENUM ('Tidak Memakai Helm', 'Tidak Memakai Masker', 'Tidak Memakai Rompi');
        END IF;
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
        link_playback VARCHAR(255)
    );
    """)

    conn.commit()
    print("Tables created successfully.")

except psycopg2.Error as e:
    print(f"Database error: {e}")

finally:
    # Close database connection
    cursor.close()
    conn.close()
