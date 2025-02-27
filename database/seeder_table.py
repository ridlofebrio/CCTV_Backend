import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random

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
    # Clear existing data
    clear_data = """
    BEGIN;
        TRUNCATE TABLE detection, ppa, cctv CASCADE;
        ALTER SEQUENCE detection_id_seq RESTART WITH 1;
        ALTER SEQUENCE ppa_id_seq RESTART WITH 1;
        ALTER SEQUENCE cctv_id_seq RESTART WITH 1;
    COMMIT;
    """
    cursor.execute(clear_data)

    # Seed CCTV data
    cctv_data = [
        ('Hikvision', 'Gedung PIP', 1, -7.205353, 112.741598, 'https://cdn-icons-png.freepik.com/512/5817/5817084.png'),
        ('Dahua', 'Dock Irian', 2, -7.204889472323035, 112.73974910298983, 'https://cdn-icons-png.freepik.com/512/5817/5817084.png'),
        ('Axis', 'Divisi Kapal Perang', 1, -7.205132962241805, 112.73865248258255, 'https://cdn-icons-png.freepik.com/512/5817/5817084.png'),
        ('Hikvision', 'Gedung PIP', 1, -7.205274124365183, 112.74220431467062, 'https://cdn-icons-png.freepik.com/512/5817/5817084.png'),
        ('Dahua', 'Dock Irian', 2,-7.20345397756675, 112.73951137680253, 'https://cdn-icons-png.freepik.com/512/5817/5817084.png'),
        ('Axis', 'Divisi Kapal Perang', 1, -7.204358729783707, 112.73798788207719, 'https://cdn-icons-png.freepik.com/512/5817/5817084.png'),
    ]
    
    insert_cctv = """
    INSERT INTO cctv (merek, gedung, lantai, latitude, longitude, gambar) 
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    cursor.executemany(insert_cctv, cctv_data)

    # Seed PPA data
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

    # Generate detection data for the last 24 hours
    detection_data = []
    now = datetime.now()
    start_time = now - timedelta(hours=10)
    
    # Generate a detection every 30 minutes
    while start_time <= now:
        for cctv_id in range(1, 4):  # For each CCTV
            # Randomly decide if we create a detection
            if random.random() < 0.7:  # 70% chance of detection
                detection = (
                    cctv_id,  
                    random.randint(1, 3),  # id_ppa
                    random.choice([True, False]),  # deteksi_jatuh
                    random.randint(0, 60),  # deteksi_overtime
                    f'Playback/violation_{start_time.strftime("%Y%m%d_%H%M%S")}.mp4',  # link_playback
                    start_time,  # timestamp
                    random.uniform(50, 95)  # confidence
                )
                detection_data.append(detection)
        
        start_time += timedelta(minutes=30)
    
    # Insert detection data
    insert_detection = """
    INSERT INTO detection (
        id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback, 
        timestamp, confidan
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    cursor.executemany(insert_detection, detection_data)

    conn.commit()
    print(f"Data seeded successfully with {len(detection_data)} detections over the last 24 hours")

except psycopg2.Error as e:
    print(f"Database error: {e}")
    conn.rollback()

finally:
    cursor.close()
    conn.close()
