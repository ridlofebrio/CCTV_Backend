# Detection System

Sistem deteksi menggunakan YOLO 11 untuk memantau keselamatan pekerja melalui CCTV.

## Fitur

- Deteksi pekerja secara real-time
- Monitoring pelanggaran K3:
  - Tidak memakai helm
  - Tidak memakai masker
  - Tidak memakai rompi
- Deteksi pekerja jatuh
- Deteksi overtime di suatu tempat
- Integrasi dengan database PostgreSQL

## Struktur Database

### Tabel CCTV
- id (Primary Key)
- merek
- gedung
- lantai 
- koordinat (POINT)

### Tabel PPA (Pelanggaran)
- id (Primary Key)
- label (ENUM):
  - Tidak Memakai Helm
  - Tidak Memakai Masker
  - Tidak Memakai Rompi

### Tabel Detection
- id (Primary Key)
- id_cctv (Foreign Key)
- id_ppa (Foreign Key)
- deteksi_jatuh
- deteksi_overtime
- link_playback

## Persyaratan Sistem

- Python >= 3.8
- OpenCV >= 4.8.0
- YOLOv8 >= 8.0.0
- PostgreSQL >= 12
- Dependencies lain ada di requirements.txt

## Instalasi

1. Clone repository ini
2. Buat virtual environment: