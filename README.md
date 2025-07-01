# Klasifikasi Penyakit Daun Menggunakan Fitur Citra dan Deep Learning

Ini adalah aplikasi desktop berbasis Python menggunakan Tkinter untuk klasifikasi penyakit pada daun. Aplikasi ini memanfaatkan teknik pemrosesan citra untuk mengekstrak fitur dari daun dan model Deep Learning (Keras) yang telah dilatih untuk memprediksi jenis penyakit.

## Fitur Utama

* **Pemrosesan Citra Daun**:
    * Penghapusan *background* otomatis menggunakan `rembg`.
    * Deteksi dan pemotongan (cropping) area daun.
    * Tampilan berbagai kanal citra: Hue, Saturation, Value, Grayscale, dan Mask Bentuk Daun.
* **Ekstraksi Fitur Citra**:
    * Mengekstrak fitur tekstur menggunakan **Gray Level Co-occurrence Matrix (GLCM)**.
    * Mengekstrak fitur warna dari kanal **HSV (Hue, Saturation, Value)**.
    * Mengekstrak fitur bentuk (misalnya, *eccentricity* dan *circularity*) menggunakan `scikit-image`.
* **Klasifikasi Penyakit**:
    * Memuat model Deep Learning (Keras) yang telah dilatih (`model9.h5`) untuk prediksi.
    * Menampilkan hasil klasifikasi penyakit (Antraknosa, Bercak Daun (PSD), Hawar Daun, Normal).
    * Menampilkan akurasi model berdasarkan dataset pengujian internal (`datatesting.xlsx`).
* **Antarmuka Pengguna Grafis (GUI)**:
    * Antarmuka yang intuitif dan mudah digunakan dibangun dengan Tkinter.
    * Visualisasi *real-time* dari langkah-langkah pemrosesan citra.

## Persyaratan Sistem

* Python 3.x
* PIP (Package Installer for Python)

## Instalasi

1.  **Kloning Repositori:**
    ```bash
    git clone [https://github.com/NamaPenggunaAnda/NamaRepositoriAnda.git](https://github.com/NamaPenggunaAnda/NamaRepositoriAnda.git)
    cd NamaRepositoriAnda
    ```
    *(Ganti `NamaPenggunaAnda` dan `NamaRepositoriAnda` dengan yang sesuai)*

2.  **Buat Lingkungan Virtual (Direkomendasikan):**
    ```bash
    python -m venv venv
    # Di Windows:
    venv\Scripts\activate
    # Di macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Jika Anda belum memiliki `requirements.txt`, buatlah dengan perintah berikut setelah instalasi manual)*:
    ```bash
    pip freeze > requirements.txt
    ```

    **Daftar Dependensi Utama (yang akan ada di `requirements.txt`):**
    * `tkinter` (biasanya sudah termasuk dengan instalasi Python standar)
    * `numpy`
    * `pandas`
    * `opencv-python` (`cv2`)
    * `XlsxWriter`
    * `scikit-learn`
    * `tensorflow` atau `keras` (tergantung versi Keras Anda)
    * `matplotlib`
    * `rembg[cpu]` (untuk penghapusan background, tambahkan `[cpu]` jika tidak ingin menggunakan GPU)
    * `scikit-image`
    * `Pillow` (`PIL`)

## Struktur Proyek

├── GUI-Perbaikan.py        # Kode utama aplikasi GUI
├── model9.h5               # File model Deep Learning yang telah dilatih
├── datatesting.xlsx        # Dataset untuk melatih LabelEncoder dan menguji akurasi model
└── README.md               # File ini


## Penggunaan

1.  **Pastikan Anda memiliki `model9.h5` dan `datatesting.xlsx` di direktori yang sama dengan `GUI-Perbaikan.py`.**
    * `model9.h5`: Model Keras yang telah dilatih untuk klasifikasi penyakit.
    * `datatesting.xlsx`: Berisi fitur dan label penyakit yang digunakan untuk inisialisasi `LabelEncoder` dan evaluasi akurasi model. Pastikan file ini memiliki kolom 'Keterangan' dengan label kelas.

2.  **Jalankan Aplikasi:**
    ```bash
    python GUI-Perbaikan.py
    ```

3.  **Langkah-langkah di Aplikasi:**
    * Klik tombol **"OPEN IMAGE"** untuk memilih gambar daun cabai (format `.png`, `.jpg`, `.jpeg`, dll.).
    * Setelah gambar dimuat, gambar daun yang sudah diproses (tanpa *background* dan terpotong) akan ditampilkan di bagian **"Normal (Diproses)"**.
    * Anda dapat melihat berbagai kanal citra dengan mengklik tombol **"HUE"**, **"SATURATION"**, **"VALUE"**, **"GRAYSCALE"**, dan **"SHAPE"**.
    * Klik tombol **"EKSTRAK CIRI"** untuk memulai proses ekstraksi fitur dari gambar yang dimuat dan melakukan klasifikasi menggunakan model. Sebuah pesan akan muncul setelah proses selesai.
    * Klik tombol **"DETEKSI PENYAKIT"** untuk melihat hasil klasifikasi penyakit.
    * Klik tombol **"LIHAT AKURASI"** untuk menampilkan akurasi model secara keseluruhan berdasarkan `datatesting.xlsx`.

## Catatan Penting

* **Model dan Data Pengujian**: Model (`model9.h5`) dan data pengujian (`datatesting.xlsx`) adalah komponen krusial. Pastikan keduanya valid dan sesuai dengan format yang diharapkan oleh kode. `datatesting.xlsx` harus berisi kolom fitur yang sama (nama dan urutan) dengan data yang digunakan untuk melatih model, serta kolom 'Keterangan' untuk label kelas.
* **Performa**: Untuk gambar beresolusi sangat tinggi, pemrosesan mungkin memakan waktu. Kode telah menyertakan *resizing* awal untuk kinerja yang lebih baik.
* **Debug**: Pesan *error* dan *print statement* telah disertakan untuk membantu *debugging* jika terjadi masalah.

## Kontribusi

Kontribusi dalam bentuk *bug reports*, *feature requests*, atau *pull requests* sangat dihargai.

---

**Dibuat dengan ❤️ oleh [Kurniadinur]**
