import os
import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
from flask import current_app
from database.database import get_db
import time
import threading
import queue
import logging
from time import sleep
import psycopg2 # type: ignore
from datetime import timedelta, datetime
from dotenv import load_dotenv
from collections import defaultdict
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load YOLO model
model_path = "Model\ppe.pt"
model = YOLO(model_path)

LABEL_MAP = {
    0: 'Hardhat',
    1: 'Mask', 
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}

CONFIDENCE_THRESHOLD = 0.5

_stream_handlers = {}
_handlers_lock = threading.Lock()

DETECTION_COOLDOWN = 5  # seconds between same violation detections
last_detection_time = defaultdict(float)


class RTSPStreamHandler:
    def __init__(self, source, model, buffer_size=30):
        self.source = source
        self.model = model
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.processed_frame = None
        self.last_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.last_access = time.time()

    def start(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.process_thread = threading.Thread(target=self._process_frames)
        self.capture_thread.daemon = True
        self.process_thread.daemon = True
        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)

    def _capture_frames(self):
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logging.error(f"Tidak dapat membuka stream RTSP: {self.source}")
                return
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            retry_count = 0
            max_retries = 5
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    logging.warning(f"Gagal membaca frame, percobaan ke-{retry_count}")
                    if retry_count > max_retries:
                        logging.error("Melebihi batas maksimum percobaan, menghentikan stream")
                        break
                    sleep(1)
                    continue
                
                retry_count = 0  # Reset counter jika berhasil
                
                with self.lock:
                    self.last_frame = frame.copy()
                    self.last_access = time.time()

                if not self.frame_buffer.full():
                    self.frame_buffer.put(frame)
                
        except Exception as e:
            logging.error(f"Error pada _capture_frames: {str(e)}")
        finally:
            if cap:
                cap.release()

    def _process_frames(self):
        while self.running:
            try:
                frame = self.frame_buffer.get(timeout=1)
                if frame is None:
                    continue
                
                processed_frame = frame.copy()
                results = self.model(frame, stream=False)
                
                # Tambahkan variabel untuk menandai ada pelanggaran
                has_violation = False
                
                for result in results:
                    if not hasattr(result, "boxes") or result.boxes is None:
                        continue
                        
                    for box in result.boxes:
                        try:
                            confidence = float(box.conf[0].item())
                            class_id = int(box.cls[0].item())
                            
                            if confidence < CONFIDENCE_THRESHOLD:
                                continue
                                
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            label = LABEL_MAP.get(class_id, f"Unknown-{class_id}")
                            
                            # Cek pelanggaran APD
                            is_violation = check_apd_violation(label)
                            if is_violation:
                                has_violation = True
                            
                            # Tentukan warna berdasarkan pelanggaran APD
                            color = (0, 0, 255) if is_violation else (0, 255, 0)
                            
                            # Gambar bounding box dan label
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                processed_frame,
                                f"{label} ({confidence:.2f})",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                            
                            # Simpan deteksi pelanggaran ke database
                            if is_violation:
                                save_detection_to_db(None, label, confidence)
                                
                        except Exception as e:
                            logging.error(f"Error processing box: {str(e)}")
                            continue
                
                # Tambahkan border merah jika ada pelanggaran
                if has_violation:
                    border_thickness = 10
                    h, w = processed_frame.shape[:2]
                    # Gambar border merah di sekeliling frame
                    cv2.rectangle(processed_frame, (0,0), (w,h), (0,0,255), border_thickness)
                    
                with self.lock:
                    self.processed_frame = processed_frame
                    self.last_access = time.time()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing frame: {str(e)}")
                continue


    def get_frame(self):
        with self.lock:
            self.last_access = time.time()
            if self.processed_frame is not None:
                return self.processed_frame
            return self.last_frame if self.last_frame is not None else None


def get_stream_handler(video_source, model):
    """Helper function untuk mendapatkan atau membuat stream handler"""
    with _handlers_lock:
        handler = _stream_handlers.get(video_source)
        if handler is None or not handler.running:
            handler = RTSPStreamHandler(video_source, model)
            handler.start()
            _stream_handlers[video_source] = handler
        return handler


def cleanup_handlers(max_idle_time=30):
    """Membersihkan handler yang tidak aktif"""
    with _handlers_lock:
        current_time = time.time()
        inactive = []
        for source, handler in _stream_handlers.items():
            if current_time - handler.last_access > max_idle_time:
                handler.stop()
                inactive.append(source)

        for source in inactive:
            del _stream_handlers[source]


def generate_frames(video_source):
    handler = get_stream_handler(video_source, model)
    
    while True:
        try:
            frame = handler.get_frame()
            if frame is None:
                continue
                
            # Encode frame ke format JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            logging.error(f"Kesalahan saat memproses frame: {str(e)}")
            break


def save_frame_with_bbox(frame, frame_count, user_id):
    """
    Menyimpan frame full dengan bounding box untuk laporan email.
    """
    try:
        timestamp = int(time.time())
        image_filename = f"fall_{user_id}_{timestamp}_{frame_count}_bbox.jpg"

        abs_image_path = os.path.join(
            current_app.config['DETECTION_IMAGES_FOLDER'], image_filename)
        rel_image_path = f"uploads/detections/{image_filename}"

        os.makedirs(os.path.dirname(abs_image_path), exist_ok=True)

        success = cv2.imwrite(abs_image_path, frame)
        if not success:
            raise Exception("Failed to save image")

        print(f"Frame saved successfully to: {abs_image_path}")

        return rel_image_path

    except Exception as e:
        print(f"Error saving frame: {str(e)}")
        return None


# Ubah fungsi save_detection_to_db
def save_detection_to_db(user_id, label, confidence):
    """
    Menyimpan data deteksi ke database sesuai schema yang ada
    """
    ppa_label = convert_label_to_ppa(label)
    if ppa_label:  # Only save if we have a valid PPA label
        connection = get_db()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback)
            VALUES (1, 
                   (SELECT id FROM ppa WHERE label = %s LIMIT 1), 
                   false, 
                   0, 
                   NULL)
        """, (ppa_label,))
        connection.commit()
        cursor.close()


def detect_and_label(frame, user_id):
    results = model(frame, stream=False) 
    for result in results:
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                label = LABEL_MAP.get(class_id, f"Unknown-{class_id}")
                
                # Cek pelanggaran dan tentukan warna
                is_violation = check_apd_violation(label)
                color = (0, 0, 255) if is_violation else (0, 255, 0)

                if check_apd_violation(label):
                    save_detection_to_db(user_id, label, confidence)

                # Gunakan warna yang sesuai
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def check_apd_violation(label):
    """
    Memeriksa pelanggaran APD berdasarkan label.
    """
    violation_labels = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
    return label in violation_labels


def process_video(input_path, frame_width=1536, frame_height=864):
    """
    Process video with person counting and PPE detection
    """
    try:
        # Database connection setup
        load_dotenv()
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cursor = conn.cursor()

        # Video capture setup
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        # Initialize variables
        frame_count = 0
        person_counts = []
        start_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Check elapsed time (10 minutes = 600 seconds)
            elapsed_time = time.time() - start_time.timestamp()
            if (elapsed_time >= 600) or cv2.waitKey(1) & 0xFF == ord('q'):
                print("Time limit reached or 'q' key pressed. Stopping capture.")
                break

            frame_count += 1
            seconds = frame_count / fps
            timestamp = str(timedelta(seconds=seconds)).split('.')[0]

            # Detect and track objects
            results = model.track(frame, persist=True)

            # Process detections
            if results and len(results) > 0:
                # Count persons and check PPE violations
                person_boxes = []
                violation_detected = False

                for box in results[0].boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    label = LABEL_MAP.get(class_id, 'Unknown')
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Track persons
                    if label == 'Person':
                        person_boxes.append(box)
                        person_id = int(box.id) if box.id is not None else 'N/A'
                        color = (255, 0, 255)  # Pink color for person detection
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_text = f"Person ID: {person_id}, Conf: {confidence:.2f}"
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Check for PPE violations
                    elif label.startswith('NO-'):
                        violation_detected = True
                        save_violation_to_db(cursor, label, confidence, timestamp)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Update person count
                person_count = len(person_boxes)
                person_counts.append(person_count)

                # Display counts and FPS
                cv2.putText(frame, f'Total Persons: {person_count}', (10, 820),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                current_fps = frame_count / (time.time() - start_time.timestamp())
                cv2.putText(frame, f'FPS: {current_fps:.2f}', (10, 850),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                # Only save if there's a violation (has id_ppa)
                if violation_detected:
                    save_violation_to_db(cursor, label, confidence, timestamp)
                    conn.commit()

            # Display the frame
            cv2.imshow('Frame', frame)

        # Calculate max consecutive count
        process_max_consecutive_count(cursor, person_counts, start_time, datetime.now())

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        cursor.close()
        conn.close()

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

# Tambah fungsi helper untuk konversi label
def convert_label_to_ppa(label):
    """
    Konversi label deteksi ke format PPA database
    """
    label_map = {
        'NO-Hardhat': 'Tidak Memakai Helm',
        'NO-Mask': 'Tidak Memakai Masker', 
        'NO-Safety Vest': 'Tidak Memakai Rompi'
    }
    return label_map.get(label)

# Update the save_violation_to_db function definition
def save_violation_to_db(cursor, label, confidence=None, timestamp=None):
    """Save PPE violation to database with cooldown"""
    global last_detection_time
    
    current_time = time.time()
    ppa_label = convert_label_to_ppa(label)
    
    if ppa_label:
        # Check if enough time has passed since last detection of this violation type
        if current_time - last_detection_time[ppa_label] >= DETECTION_COOLDOWN:
            cursor.execute("""
            INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback)
            VALUES (1, 
                   (SELECT id FROM ppa WHERE label = %s LIMIT 1),
                   false,
                   0,
                   NULL)
            """, (ppa_label,))
            # Update last detection time for this violation type
            last_detection_time[ppa_label] = current_time

# def process_max_consecutive_count(cursor, person_counts, start_time, end_time):
#     """Process and save maximum consecutive person count"""
#     if person_counts:
#         max_person_count = None
#         consecutive_count = 1

#         for i in range(1, len(person_counts)):
#             if person_counts[i] == person_counts[i - 1]:
#                 consecutive_count += 1
#             else:
#                 consecutive_count = 1

#             if consecutive_count > 15:
#                 if max_person_count is None or person_counts[i] > max_person_count:
#                     max_person_count = person_counts[i]

#         if max_person_count is not None:
#             cursor.execute("""
#             INSERT INTO person_count_max (max_value, start_time, end_time, cctv_id)
#             VALUES (%s, %s, %s, 1)
#             """, (max_person_count, start_time, end_time))
#             print(f"Maximum person count that appeared more than 15 times consecutively: {max_person_count}")
#         else:
#             print("No person count appeared more than 15 times consecutively.")

if __name__ == "__main__":
    video_path = 'video-test\huuman.mp4'
    process_video(video_path)

