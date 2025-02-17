import psycopg2 # type: ignore
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
    # Wrap truncates in a transaction
    clear_data = """
    BEGIN;
        TRUNCATE TABLE detection, ppa, cctv CASCADE;
        ALTER SEQUENCE detection_id_seq RESTART WITH 1;
        ALTER SEQUENCE ppa_id_seq RESTART WITH 1;
        ALTER SEQUENCE cctv_id_seq RESTART WITH 1;
    COMMIT;
    """
    
    cursor.execute(clear_data)

    # Seed data CCTV
    cctv_data = [
        ('Hikvision', 'Gedung A', 1, '(106.8456,-6.2088)'),
        ('Dahua', 'Gedung B', 2, '(106.8457,-6.2089)'),
        ('Axis', 'Gedung C', 1, '(106.8458,-6.2090)'),
    ]
    
    insert_cctv = """
    INSERT INTO cctv (merek, gedung, lantai, koordinat) 
    VALUES (%s, %s, %s, POINT(%s));
    """
    cursor.executemany(insert_cctv, cctv_data)

    # Seed data PPA
    ppa_data = [
        ('Tidak Memakai Helm',),
        ('Tidak Memakai Masker',),
        ('Tidak Memakai Rompi',),
    ]
    
    insert_ppa = """
    INSERT INTO ppa (label) 
    VALUES (%s);
    """
    cursor.executemany(insert_ppa, ppa_data)

    # Seed data Detection
    detection_data = [
        (1, 1, True, 30, 'https://storage/video1.mp4'),
        (2, 2, False, 45, 'https://storage/video2.mp4'),
        (3, 3, True, 60, 'https://storage/video3.mp4'),
    ]
    
    insert_detection = """
    INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback) 
    VALUES (%s, %s, %s, %s, %s);
    """
    cursor.executemany(insert_detection, detection_data)

    conn.commit()
    print("Data berhasil di-seed ke database.")

except psycopg2.Error as e:
    print(f"Database error: {e}")

finally:
    cursor.close()
    conn.close()
