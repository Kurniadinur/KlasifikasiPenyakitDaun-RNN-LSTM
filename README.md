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
