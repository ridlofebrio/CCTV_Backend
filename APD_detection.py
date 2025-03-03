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
import threading
from copy import deepcopy

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
load_dotenv()

# Check CUDA availability and set device
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'
print(f"Using device: {device}")

if cuda_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU. Detection will be slower.")

# Load YOLO model
model_path = "Model/best (3).pt"
model = YOLO(model_path)
model.to(device)  # Use selected device

# Update LABEL_MAP to remove 'Tidak Memakai Rompi'
LABEL_MAP = {
    0: 'Memakai Helm',      # Hardhat
    1: 'Memakai Masker',    # Mask
    2: 'Tidak Memakai Helm', # NO-Hardhat
    3: 'Tidak Memakai Masker', # NO-Mask
    4: None,  # Ignore NO-Safety Vest
    5: None,
    6: 'Safety Cone',
    7: 'Memakai Rompi',     # Safety Vest
    8: 'machinery',
    9: 'vehicle'
}
CONFIDENCE_THRESHOLD = 0.1
DETECTION_COOLDOWN = 70  # seconds
last_detection_time = defaultdict(float)

RECORD_DURATION = 20  # seconds
RECORD_FPS = 30
PLAYBACK_FOLDER = "Playback"
LOCALHOST_URL = "playback"  # Base URL for video access


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
        # Skip database insert if video_path is None
        if video_path is None:
            logging.warning(f"Skipping database insert - No video path for violation: {ppa_label}")
            return

        # Get ppa_id based on label
        cursor.execute("""
            SELECT id FROM ppa WHERE label = %s
        """, (ppa_label,))
        result = cursor.fetchone()
        
        if not result:
            logging.error(f"PPA label not found in database: {ppa_label}")
            return
            
        ppa_id = result[0]

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
            video_path,
            confidence
        ))

        logging.info(
            f"APD violation saved: Type={ppa_label}, Confidence={confidence}%, Video={video_path}")

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Error saving violation: {str(e)}")


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
        
        # Try multiple codecs until one works
        codecs_to_try = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # More compatible
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Widely supported
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # Try avc1 if available
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'))   # Last resort
        ]
        
        out = None
        for codec_name, fourcc in codecs_to_try:
            try:
                test_out = cv2.VideoWriter(temp_path, fourcc, RECORD_FPS, 
                                        (frame_width, frame_height))
                if test_out.isOpened():
                    out = test_out
                    logging.info(f"Using codec: {codec_name}")
                    break
                else:
                    test_out.release()
            except Exception as codec_error:
                logging.warning(f"Codec {codec_name} failed: {codec_error}")
        
        if out is None or not out.isOpened():
            logging.error("Could not initialize any video codec")
            return None
            
        frames_to_capture = RECORD_DURATION * RECORD_FPS
        frames_captured = 0
        
        # Create window for recording
        cv2.namedWindow('Recording Violation', cv2.WINDOW_NORMAL)
        
        while frames_captured < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with detections - handle both CPU and GPU
            if cuda_available:
                with torch.amp.autocast('cuda'):
                    results = model(frame, stream=False)
            else:
                # For CPU, don't use autocast
                results = model(frame, stream=False)
            
            # Add detections to frame
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    current_label = LABEL_MAP.get(class_id, 'Unknown')
                    
                    if confidence < CONFIDENCE_THRESHOLD or current_label is None:
                        continue
                    
                    # Only show Person and violations
                    if current_label is not None and (current_label == 'Person' or current_label.startswith('Tidak')):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        color = (255, 0, 255) if current_label == 'Person' else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, current_label, 
                                  (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add recording status to frame
            cv2.putText(frame, f"Recording violation: {label}", 
                      (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {frames_captured}/{frames_to_capture}", 
                      (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out.write(frame)
            frames_captured += 1
            
            # Display recording progress
            cv2.imshow('Recording Violation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        out.release()
        
        # Try FFmpeg conversion
        try:
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-i', temp_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-movflags', '+faststart',
                '-y', final_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            os.remove(temp_path)  # Remove temporary file
        except Exception as e:
            logging.error(f"FFmpeg conversion failed: {e}")
            # If FFmpeg fails, use the original file
            try:
                os.rename(temp_path, final_path)
            except Exception as rename_error:
                logging.error(f"Failed to rename temp file: {rename_error}")
        
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        # Close recording window
        cv2.destroyWindow('Recording Violation')
        
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
            
        # For CPU usage, consider using a smaller frame size to improve performance
        if not cuda_available:
            frame_width = min(frame_width, 960)  # Lower resolution for CPU
            frame_height = min(frame_height, 540)
            logging.info(f"Using reduced resolution ({frame_width}x{frame_height}) for CPU processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))

            # Run detection with appropriate acceleration
            if cuda_available:
                with torch.amp.autocast('cuda'):
                    results = model(frame, stream=False)
            else:
                # For CPU, don't use autocast
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

                    # Skip if label is None (Tidak Memakai Rompi) or confidence too low
                    if label is None or confidence < CONFIDENCE_THRESHOLD:
                        continue

                    # Only process Person and specific violations (not rompi)
                    if label is not None and (label == 'Person' or (label.startswith('Tidak') and label != 'Tidak Memakai Rompi')):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        if label == 'Person':
                            person_count += 1
                            # No need to draw since we're not displaying
                        elif label != 'Tidak Memakai Rompi':  # Extra check
                            violation_detected = True
                            # Handle violation recording and database
                            current_time = time.time()
                            if current_time - last_detection_time[label] >= DETECTION_COOLDOWN:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                video_path = record_violation_video(cap, label, timestamp)
                                
                                # Only save to database if we have a valid video path
                                if video_path:
                                    save_violation_to_db(cursor, label, confidence * 100, video_path)
                                    connection.commit()
                                    last_detection_time[label] = current_time
                                    logging.info(f"Violation detected and recorded: {label}")
                                else:
                                    logging.warning(f"Video recording failed for violation: {label}")

            # Only show status info in terminal, don't display the main detection window
            if violation_detected:
                logging.info(f"Violation detected: {label}, Processing...")
            
            # Check for quit key without showing window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        connection = get_db()
        cursor = connection.cursor()
        video_path = ""
        process_video(cursor, video_path)
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

