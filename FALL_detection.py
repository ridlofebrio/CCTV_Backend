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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Check CUDA availability
print(f"Using CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Load YOLO model
model_path = "Model/fall2.pt"
model = YOLO(model_path)
model.to('cuda')

LABEL_MAP = {
    0: "Jatuh",
    1: "Normal"
}
CONFIDENCE_THRESHOLD = 0.71
DETECTION_COOLDOWN = 60  # seconds
last_detection_time = defaultdict(float)

RECORD_DURATION = 20  # seconds
RECORD_FPS = 20
PLAYBACK_FOLDER = "Playback"


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


def record_fall_video(cap, label, timestamp):
    """Records video for specified duration when fall is detected"""
    try:
        # Create filename with timestamp
        video_filename = f"Fall_{label}_{timestamp}.mp4"
        video_path = os.path.join(PLAYBACK_FOLDER, video_filename)

        # Ensure directory exists
        os.makedirs(PLAYBACK_FOLDER, exist_ok=True)

        # Get original video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, RECORD_FPS,
                            (frame_width, frame_height))

        # Calculate frames to capture
        frames_to_capture = RECORD_DURATION * RECORD_FPS
        frames_captured = 0
        start_time = time.time()

        # Start recording
        while frames_captured < frames_to_capture and (time.time() - start_time) < RECORD_DURATION:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect falls in each frame of recording
            results = model(frame, stream=False)

            for result in results:
                if hasattr(result, "boxes") and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        label = LABEL_MAP.get(class_id)

                        if confidence > CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Add label to frame
            cv2.putText(frame, f"Fall Detected: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(frame)
            frames_captured += 1

            # Display recording progress
            cv2.imshow('Recording Fall', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        return video_path

    except Exception as e:
        logging.error(f"Error recording fall video: {str(e)}")
        return None


# Modify the process_fall_detection function
def process_fall_detection(input_path, cctv_id=1, frame_width=640, frame_height=480):
    """Processes video or RTSP stream to detect fall events"""
    try:
        # Video/RTSP capture setup
        if input_path.startswith('rtsp://'):
            # RTSP specific settings
            cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Minimize latency
        else:
            cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))

            # Detect falls
            results = model(frame, stream=False)

            for result in results:
                if hasattr(result, "boxes") and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        label = LABEL_MAP.get(class_id)

                        if confidence > CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Save to database if fall is detected
                            if label == "Jatuh":
                                current_time = time.time()
                                if current_time - last_detection_time[cctv_id] >= DETECTION_COOLDOWN:
                                    is_fall = True
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    video_path = record_fall_video(cap, label, timestamp)
                                    save_fall_detection_to_db(cctv_id, is_fall, confidence, video_path)
                                    last_detection_time[cctv_id] = current_time
                                else:
                                    is_fall = False

            cv2.imshow('Fall Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error processing video: {e}")


if __name__ == "__main__":
    video_path = "video-test/fall5.mp4"
    # video_path = "rtsp://username:password@your_rtsp_stream_address"  # Replace with your RTSP stream URL
    process_fall_detection(video_path)