import os
import cv2
from ultralytics import YOLO
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch  # Import torch untuk memeriksa ketersediaan CUDA

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Periksa ketersediaan CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Konfigurasi model YOLO
MODEL_PATH = "Model/People.pt"
model = YOLO(MODEL_PATH)
model.to(device)  # Gunakan CUDA jika tersedia

# Inisialisasi DeepSORT tracker
tracker = DeepSort(
    max_age=30, n_init=3, nn_budget=100, use_cuda=(device == "cuda")
)  # Gunakan CUDA jika tersedia

# Konfigurasi database
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Konfigurasi deteksi
DETECTION_THRESHOLD = 0.5  # Ambang kepercayaan deteksi
OVERTIME_THRESHOLD = 10  # Waktu maksimum (dalam detik) sebelum dianggap pelanggaran
RECORD_DURATION = 20  # Durasi rekaman (dalam detik)
RECORD_FPS = 30
PLAYBACK_FOLDER = "E:\\Magang\\Project\\Backend_CCTV\\Backend_CCTV\\Playback"


def save_violation_to_db(cursor, id_cctv, overtime_duration, video_path):
    """
    Menyimpan data pelanggaran ke database.
    """
    cursor.execute(
        """
    INSERT INTO detection (id_cctv, id_ppa, deteksi_jatuh, deteksi_overtime, link_playback)
    VALUES (%s, NULL, FALSE, %s, %s)
    """,
        (id_cctv, overtime_duration, video_path),
    )


def record_detection_video(cap, timestamp, initial_detection_duration=0):
    """
    Merekam hasil proses video (deteksi orang) dan menyimpannya ke file.
    """
    try:
        # Buat nama file video
        video_filename = f"Detection_{timestamp}.mp4"
        video_path = os.path.join(PLAYBACK_FOLDER, video_filename)

        # Pastikan folder playback ada
        os.makedirs(PLAYBACK_FOLDER, exist_ok=True)

        # Ambil properti video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inisialisasi video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            video_path, fourcc, RECORD_FPS, (frame_width, frame_height)
        )

        # Hitung jumlah frame yang perlu direkam
        frames_to_capture = RECORD_DURATION * RECORD_FPS
        frames_captured = 0

        # Mulai merekam
        while frames_captured < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi orang menggunakan YOLO
            results = model(frame, conf=DETECTION_THRESHOLD)

            # Konversi hasil deteksi ke format yang sesuai untuk DeepSORT
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, "person"))

            # Update tracker dengan deteksi terbaru
            tracks = tracker.update_tracks(detections, frame=frame)

            # Proses setiap objek yang dilacak
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                bbox = (x1, y1, x2 - x1, x2 - y1)

                # Hitung durasi deteksi
                detection_duration = initial_detection_duration + (
                    frames_captured / RECORD_FPS
                )

                # Tentukan warna berdasarkan durasi deteksi
                color = (
                    (0, 255, 0)
                    if detection_duration <= OVERTIME_THRESHOLD
                    else (0, 0, 255)
                )

                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Tambahkan label, confidence score, dan timer
                cv2.putText(
                    frame,
                    f"ID {track_id} - {str(timedelta(seconds=int(detection_duration)))}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Tulis frame ke video
            out.write(frame)
            frames_captured += 1

            # Tampilkan progress rekaman
            cv2.imshow("Recording Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        return video_path

    except Exception as e:
        logging.error(f"Error recording detection video: {str(e)}")
        return None


def process_video(input_path, id_cctv=1):
    """
    Memproses video untuk mendeteksi orang dan menyimpan hasil deteksi ke video.
    """
    try:
        # Koneksi ke database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Buka video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        # Inisialisasi variabel
        start_time = datetime.now()
        detection_start_time = None
        overtime_duration = 0
        violation_bbox = None
        detection_duration = 0  # Initialize detection_duration
        recording = False  # Flag untuk menandai apakah sedang merekam
        video_path = None  # Path video hasil rekaman

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Deteksi orang menggunakan YOLO
            results = model(frame, conf=DETECTION_THRESHOLD)

            # Konversi hasil deteksi ke format yang sesuai untuk DeepSORT
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, "person"))

            # Update tracker dengan deteksi terbaru
            tracks = tracker.update_tracks(detections, frame=frame)

            # Proses setiap objek yang dilacak
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                bbox = (x1, y1, x2 - x1, y2 - y1)

                if detection_start_time is None:
                    detection_start_time = datetime.now()
                else:
                    # Hitung durasi deteksi (diperlambat 30 kali)
                    detection_duration = (
                        datetime.now() - detection_start_time
                    ).total_seconds() / 25
                    if detection_duration > OVERTIME_THRESHOLD:
                        overtime_duration = int(detection_duration)
                        violation_bbox = bbox
                        # Mulai merekam jika belum merekam
                        if not recording:
                            video_path = record_detection_video(
                                cap,
                                datetime.now().strftime("%Y%m%d_%H%M%S"),
                                detection_duration,
                            )
                            recording = True
                        # Simpan ke database
                        save_violation_to_db(
                            cursor, id_cctv, overtime_duration, video_path
                        )
                        conn.commit()
                        logging.info(
                            f"Pelanggaran terdeteksi: Melebihi waktu {overtime_duration} detik"
                        )
                        detection_start_time = None  # Reset timer

                # Gambar kotak, ID, dan timer
                color = (
                    (0, 255, 0)
                    if detection_duration <= OVERTIME_THRESHOLD
                    else (0, 0, 255)
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} - {str(timedelta(seconds=int(detection_duration)))}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Tampilkan frame
            cv2.imshow("People Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Bersihkan
        cap.release()
        cv2.destroyAllWindows()
        cursor.close()
        conn.close()

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    video_path = "video-test/test.mp4"
    process_video(video_path)
