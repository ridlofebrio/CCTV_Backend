# APD Detection System

APD Detection System menggunakan YOLOv11 untuk memantau keselamatan pekerja melalui CCTV dan mendeteksi pelanggaran APD (Alat Pelindung Diri).

## 🚀 Fitur
- **Deteksi Pekerja Secara Real-time**
- **Pemantauan Pelanggaran APD:**
  - Tidak Memakai Helm
  - Tidak Memakai Masker
  - Tidak Memakai Rompi
- **Deteksi Jatuh** 
- **Deteksi Lembur** 
- **Perekaman Video Pelanggaran**
- **Integrasi dengan Database PostgreSQL**
- **Akselerasi GPU dengan CUDA**
- **Integrasi Cloudinary untuk Penyimpanan Video (Opsional)**

---

## 📂 Struktur Proyek
```
.
├── .env                 # Variabel lingkungan (database, API key, dll.)
├── .gitignore           # File yang diabaikan oleh Git
├── APD_detection.py     # Skrip utama untuk deteksi APD dan pemrosesan video
├── database/            # Direktori untuk skrip database
│   ├── __pycache__/     # Direktori cache Python
│   ├── create_table.py  # Skrip untuk membuat tabel database
│   ├── database.py      # Skrip untuk koneksi database
│   └── seeder_table.py  # Skrip untuk seeding data awal ke database
├── Model/               # Direktori untuk model 
│   └── model.pt         # Model deteksi  
├── Playback/            # Direktori penyimpanan video pelanggaran
│   ├── Pelanggaran_Tidak Memakai Helm_20250219_101154.mp4
│   ├── ...
├── README.md             # Dokumentasi proyek
├── requirements.txt      # Daftar dependensi Python
├── run.py                # Skrip untuk menjalankan sistem deteksi
└── video-test/           # Direktori untuk video uji coba
    └── video.mp4         # Contoh video uji coba
```

---

## 📊 Struktur Database

### **🛡️ Table: cctv**
| Kolom        | Tipe Data            | Deskripsi        |
|-------------|---------------------|----------------|
| id          | SERIAL PRIMARY KEY  | ID CCTV        |
| merek       | VARCHAR(100)        | Merek CCTV     |
| gedung      | VARCHAR(100)        | Nama gedung    |
| lantai      | INT                 | Nomor lantai   |
| latitude    | DOUBLE PRECISION    | Koordinat latitude |
| longitude   | DOUBLE PRECISION    | Koordinat longitude |
| gambar      | VARCHAR(255)        | URL gambar CCTV |

### **🚦 Table: ppa** (Jenis Pelanggaran)
| Kolom | Tipe Data | Deskripsi |
|------|---------|-----------|
| id   | SERIAL PRIMARY KEY | ID Pelanggaran |
| label | ENUM('Tidak Memakai Helm', 'Tidak Memakai Masker', 'Tidak Memakai Rompi') | Jenis pelanggaran APD |

### **📋 Table: detection**
| Kolom          | Tipe Data           | Deskripsi                      |
|--------------|-------------------|--------------------------------|
| id           | SERIAL PRIMARY KEY | ID Deteksi                    |
| id_cctv      | INT REFERENCES cctv(id) | ID CCTV (Foreign Key)       |
| id_ppa       | INT REFERENCES ppa(id)  | ID Pelanggaran (Foreign Key) |
| deteksi_jatuh| BOOLEAN           | Flag deteksi jatuh             |
| deteksi_overtime | INT            | Durasi lembur (menit)         |
| link_playback | VARCHAR(255)     | Link video pelanggaran         |
| timestamp    | TIMESTAMP         | Waktu deteksi                  |
| confidan     | DOUBLE PRECISION  | Skor kepercayaan deteksi       |

---

## ⚙️ Persyaratan Sistem
- **Python** >= 3.8
- **OpenCV** >= 4.8.0
- **YOLOv11** >= 8.0.0
- **PyTorch** >= 2.0.0 (Dengan dukungan CUDA untuk GPU)
- **PostgreSQL** >= 12
- Dependensi lainnya tersedia di `requirements.txt`

---

## 📌 Instalasi
### **1️⃣ Clone repository**
```sh
 git clone <https://github.com/ridlofebrio/CCTV_Backend.git>
 cd APD-Detection-System
```

### **2️⃣ Buat Virtual Environment**
```sh
python -m venv venv
```

### **3️⃣ Aktifkan Virtual Environment**
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **MacOS/Linux:**
  ```sh
  source venv/bin/activate
  ```

### **4️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **5️⃣ Konfigurasi Environment Variables**
Buat file `.env` di root proyek dengan format berikut:
```env
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=your_database_port
```

### **6️⃣ Jalankan Sistem**
```sh
python run.py
```

Sistem akan:
- Membuat dan menginisialisasi database.
- Menjalankan deteksi APD secara real-time.

---

## 🔹 Catatan Penting
✅ Pastikan server PostgreSQL berjalan dengan benar.
✅ Model YOLO (`.pt`) harus diletakkan di direktori `Model`.
✅ Video uji coba bisa ditempatkan di `video-test`.
✅ Sistem akan merekam video pelanggaran ke dalam folder `Playback`.
✅ Integrasi Cloudinary bersifat opsional. Jika ingin menggunakannya, atur kredensial di file `.env`.

---