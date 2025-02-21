import os
import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore
from database.database import get_db
import time
import logging
import psycopg2  # type: ignore
from datetime import datetime
from dotenv import load_dotenv
import torch  # type: ignore
from collections import defaultdict

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name%s - %(levelname%s - %(message)s'
)

# Load environment variables
load_dotenv()

# Check CUDA availability
print(f"Using CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Load YOLO model
model_path = "Model/ppe.pt"
model = YOLO(model_path)
model.to('cuda')

# Update LABEL_MAP to match PPE violations
LABEL_MAP = {
    0: 'Memakai Helm',      # Hardhat
    1: 'Memakai Masker',    # Mask
    2: 'Tidak Memakai Helm', # NO-Hardhat
    3: 'Tidak Memakai Masker', # NO-Mask
    4: 'Tidak Memakai Rompi',  # NO-Safety Vest
    5: 'Person',
    6: 'Safety Cone',
    7: 'Memakai Rompi',     # Safety Vest
    8: 'machinery',
    9: 'vehicle'
}
CONFIDENCE_THRESHOLD = 0.5
DETECTION_COOLDOWN = 70  # seconds
last_detection_time = defaultdict(float)

RECORD_DURATION = 20  # seconds
RECORD_FPS = 30
PLAYBACK_FOLDER = "Playback"
LOCALHOST_URL = "http://localhost:5000/playback"  # Base URL for video access


def save_fall_detection_to_db(cctv_id, is_fall, confidence, video_path=None):
    """Saves fall detection data to the database"""
    try:
        connection = get_db()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback, timestamp, confidan)
            VALUES (%s, NULL, %s, 0, %s, CURRENT_TIMESTAMP, %s)
        """, (cctv_id, is_fall, video_path, confidence))

        connection.commit()
        cursor.close()
        logging.info(
            f"Fall detection saved to database: Fall={is_fall}, Confidence={confidence}, Video={video_path}")

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        if connection:
            connection.close()


def save_violation_to_db(cursor, ppa_label, confidence, video_path=None):
    """Saves APD violation to database with web-accessible video path"""
    try:
        # Get ppa_id based on label
        cursor.execute("""
            SELECT id FROM ppa WHERE label = %s
        """, (ppa_label,))
        ppa_id = cursor.fetchone()[0]

        # Insert detection record with web URL
        cursor.execute("""
            INSERT INTO detection (
                id_cctv, 
                id_ppa, 
                deteksi_jatuh, 
                deteksi_overtime, 
                link_playback, 
                timestamp, 
                confidan
            )
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s)
        """, (
            1,  # Default CCTV ID
            ppa_id,
            False,  # Not a fall detection
            0,      # No overtime
            video_path,  # Now contains web URL
            confidence
        ))

        logging.info(
            f"APD violation saved: Type={ppa_label}, Confidence={confidence}%, Video={video_path}")

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")


def record_violation_video(cap, label, timestamp):
    """Records video for specified duration when violation is detected using H.264 codec"""
    try:
        # Create filename with timestamp
        video_filename = f"Pelanggaran_{label}_{timestamp}.mp4"
        temp_path = os.path.join(PLAYBACK_FOLDER, f"temp_{timestamp}.mp4")
        final_path = os.path.join(PLAYBACK_FOLDER, video_filename)
        web_path = f"{LOCALHOST_URL}/{video_filename}"  # URL for web access
        
        # Ensure directory exists
        os.makedirs(PLAYBACK_FOLDER, exist_ok=True)
        
        # Store current position
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Get original video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer with platform-specific codec
        if os.name == 'nt':  # Windows
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:  # Linux/Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            
        out = cv2.VideoWriter(temp_path, fourcc, RECORD_FPS, 
                            (frame_width, frame_height))
        
        frames_to_capture = RECORD_DURATION * RECORD_FPS
        frames_captured = 0
        
        while frames_captured < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with detections
            with torch.amp.autocast('cuda'):
                results = model(frame, stream=False)
            
            # Add detections to frame
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    current_label = LABEL_MAP.get(class_id, 'Unknown')
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Only show Person and violations
                    if current_label == 'Person' or current_label.startswith('Tidak'):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        if current_label == 'Person':
                            color = (255, 0, 255)  # Pink
                        else:
                            color = (0, 0, 255)    # Red
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, current_label, 
                                  (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
            frames_captured += 1
            
            # Display recording progress
            cv2.imshow('Recording Violation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        out.release()
        
        # Convert to H.264 using FFmpeg
        try:
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-i', temp_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-movflags', '+faststart',
                '-y',  # Overwrite output file if it exists
                final_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            os.remove(temp_path)  # Remove temporary file
        except Exception as e:
            logging.error(f"FFmpeg conversion failed: {e}")
            # If FFmpeg fails, use the original file
            os.rename(temp_path, final_path)
        
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        # Return web accessible path
        return web_path
        
    except Exception as e:
        logging.error(f"Error recording violation video: {str(e)}")
        return None


def process_video(cursor, input_path, frame_width=1536, frame_height=864):
    """Process video with PPE detection"""
    try:
        # Get connection from cursor for commits
        connection = cursor.connection
        
        # Video/RTSP capture setup
        if input_path.startswith('rtsp://'):
            cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        else:
            cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open input source: {input_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))

            # Run detection with CUDA acceleration
            with torch.amp.autocast('cuda'):
                results = model(frame, stream=False)

            person_count = 0
            violation_detected = False

            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    label = LABEL_MAP.get(class_id, 'Unknown')

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    # Only show Person and Violation detections
                    if label == 'Person' or label.startswith('Tidak'):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        if label == 'Person':
                            person_count += 1
                            # Draw person detection in pink
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame, "Person", 
                                      (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        else:
                            violation_detected = True
                            # Draw violation in red
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, label, 
                                      (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Handle violation recording and database
                            current_time = time.time()
                            if current_time - last_detection_time[label] >= DETECTION_COOLDOWN:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                video_path = record_violation_video(cap, label, timestamp)
                                save_violation_to_db(cursor, label, confidence * 100, video_path)
                                connection.commit()
                                last_detection_time[label] = current_time

            # Display frame
            cv2.imshow('APD Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        connection = get_db()
        cursor = connection.cursor()
        video_path = "video-test/huuman.mp4"
        process_video(cursor, video_path)
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

