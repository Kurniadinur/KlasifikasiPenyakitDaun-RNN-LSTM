import tkinter as tk
from tkinter import filedialog, messagebox # Mengimpor messagebox secara eksplisit
import numpy as np
import pandas as pd
import cv2
import xlsxwriter as xw
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from PIL import Image

# --- Variabel Global (Dimuat sekali untuk efisiensi) ---
# Menginisialisasi dengan None atau nilai default
fileImage = None
img = None
img_HSV = None
mask = None
grayscale = None
H = None
S = None
V = None
hasil_klasifikasi = None # Akan menyimpan hasil klasifikasi
akurasi = None # Akan menyimpan akurasi model

# Memuat model dan encoder secara global untuk menghindari pemuatan ulang setiap kali dipanggil
model = None
enc = None
try:
    model = load_model("model9.h5")
    enc = LabelEncoder()
    # Melatih encoder dengan data dari datatesting.xlsx jika tersedia
    # Ini mengasumsikan kolom 'Keterangan' ada di datatesting.xlsx
    # Sangat penting untuk melatih encoder dengan semua label yang mungkin ditemui.
    # Untuk aplikasi nyata, pertimbangkan untuk melatih encoder pada set pelatihan lengkap Anda.
    datatesting_initial = pd.read_excel("datatesting.xlsx")
    if 'Keterangan' in datatesting_initial.columns:
        enc.fit(datatesting_initial['Keterangan'].values)
    else:
        print("Peringatan: Kolom 'Keterangan' tidak ditemukan di datatesting.xlsx. LabelEncoder mungkin tidak sepenuhnya diinisialisasi.")
except Exception as e:
    print(f"Error saat memuat model atau menginisialisasi LabelEncoder: {e}")
    messagebox.showerror("Error Startup", f"Gagal memuat model atau inisialisasi: {e}\nPastikan 'model9.h5' dan 'datatesting.xlsx' ada dan valid.")
    # Dalam aplikasi nyata, Anda mungkin ingin menonaktifkan tombol yang relevan di sini

# --- Fungsi GUI ---

def openImage():
    global fileImage, img, img_HSV, mask, grayscale, H, S, V, hasil_klasifikasi, akurasi

    fileImage = filedialog.askopenfilename(
        title="Pilih Gambar Daun",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    if not fileImage:
        return # Pengguna membatalkan

    try:
        input_image = Image.open(fileImage)
        a, b = input_image.size
        # Mengubah ukuran gambar untuk kinerja yang lebih baik jika terlalu besar, tetapi pertahankan rasio aspek
        max_dim = 600
        if a > max_dim or b > max_dim:
            if a > b:
                input_image = input_image.resize((max_dim, int(b * max_dim / a)), Image.LANCZOS)
            else:
                input_image = input_image.resize((int(a * max_dim / b), max_dim), Image.LANCZOS)

        rimg = remove(input_image)
        img_np = np.array(rimg)

        # Menangani transparansi: mengatur piksel transparan menjadi putih
        change = img_np[:, :, 3] == 0
        img_np[change] = [255, 255, 255, 255]
        img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB) # Mengonversi ke RGB karena OpenCV lebih suka 3 saluran

        # Masking untuk isolasi daun
        tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Menggunakan metode OTSU untuk thresholding jika gambar memiliki variasi yang cukup
        _, initial_mask = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.dilate(initial_mask.copy(), None, iterations=5) # Mengurangi iterasi sedikit
        mask = cv2.erode(mask.copy(), None, iterations=5) # Mengurangi iterasi sedikit

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL untuk kontur luar
        if contours:
            selected_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(selected_contour)
            # Menambahkan sedikit buffer ke bounding box untuk memastikan seluruh daun tertangkap
            buffer = 10
            x_b = max(0, x - buffer)
            y_b = max(0, y - buffer)
            w_b = min(img.shape[1], x + w + buffer) - x_b
            h_b = min(img.shape[0], y + h + buffer) - y_b

            mask = mask[y_b:y_b+h_b, x_b:x_b+w_b] # Memotong mask ke buffered bounding box
            img = img[y_b:y_b+h_b, x_b:x_b+w_b] # Memotong gambar asli (img) ke bounding box yang sama

            # Memastikan mask tidak kosong setelah pemotongan
            if mask.size == 0:
                raise ValueError("Mask kosong setelah pemotongan. Deteksi daun mungkin gagal.")
        else:
            print("Tidak ada kontur signifikan yang ditemukan. Melanjutkan dengan gambar penuh untuk ekstraksi fitur (mungkin kurang akurat).")
            # Jika tidak ada kontur, gunakan gambar yang telah diproses penuh dan mask putih penuh (atau tangani sebagai error)
            # Untuk saat ini, asumsikan `img` dan `grayscale` baik-baik saja, tetapi `mask` harus valid.
            mask = np.ones_like(tmp) * 255 # Membuat mask putih penuh jika tidak ada daun yang terdeteksi dengan jelas

        # Grayscale dan HSV untuk ekstraksi fitur
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # Menampilkan gambar normal
        ax1.clear()
        ax1.imshow(img)
        ax1.set_title("Normal (Diproses)")
        ax1.axis('off') # Sembunyikan sumbu untuk tampilan gambar yang lebih bersih
        canvas1.draw()

        # Membersihkan plot lain dan mereset output ketika gambar baru dibuka
        clear_all_plots()
        output_classification_label.config(text="Belum Klasifikasi")
        output_accuracy_label.config(text="N/A")
        hasil_klasifikasi = None
        akurasi = None

    except Exception as e:
        print(f"Error saat membuka atau memproses gambar: {e}")
        messagebox.showerror("Error Pemrosesan Gambar", f"Gagal memuat atau memproses gambar: {e}")
        # Mereset variabel global pada error untuk mencegah penggunaan data yang rusak
        fileImage, img, img_HSV, mask, grayscale, H, S, V = None, None, None, None, None, None, None


def clear_all_plots():
    """Membersihkan semua plot tampilan gambar kecuali yang pertama (Normal)."""
    for ax in [ax2, ax3, ax4, ax5, ax6]:
        ax.clear()
        ax.set_title("")
        ax.axis('off')
    canvas2.draw()
    canvas3.draw()
    canvas4.draw()
    canvas5.draw()
    canvas6.draw()

def Hue():
    if H is not None:
        ax2.clear()
        ax2.imshow(H, cmap='hsv') # Saluran Hue sering divisualisasikan dengan colormap 'hsv'
        ax2.set_title("Saluran Hue")
        ax2.axis('off')
        canvas2.draw()
    else:
        messagebox.showwarning("Peringatan", "Gambar belum dimuat atau diproses.")

def Saturation():
    if S is not None:
        ax3.clear()
        ax3.imshow(S, cmap='gray') # Saturasi bisa grayscale
        ax3.set_title("Saluran Saturasi")
        ax3.axis('off')
        canvas3.draw()
    else:
        messagebox.showwarning("Peringatan", "Gambar belum dimuat atau diproses.")

def Value():
    if V is not None:
        ax4.clear()
        ax4.imshow(V, cmap='gray') # Value bisa grayscale
        ax4.set_title("Saluran Value")
        ax4.axis('off')
        canvas4.draw()
    else:
        messagebox.showwarning("Peringatan", "Gambar belum dimuat atau diproses.")

def Grayscales():
    if grayscale is not None:
        ax5.clear()
        ax5.imshow(grayscale, cmap="gray")
        ax5.set_title("Gambar Grayscale")
        ax5.axis('off')
        canvas5.draw()
    else:
        messagebox.showwarning("Peringatan", "Gambar belum dimuat atau diproses.")

def Shape():
    if mask is not None:
        ax6.clear()
        ax6.imshow(mask, cmap="gray")
        ax6.set_title("Mask Daun (Bentuk)")
        ax6.axis('off')
        canvas6.draw()
    else:
        messagebox.showwarning("Peringatan", "Gambar belum dimuat atau diproses.")

def ekstrak_ciri():
    global hasil_klasifikasi, akurasi, model, enc

    if img is None or grayscale is None or mask is None:
        messagebox.showwarning("Peringatan", "Gambar belum dimuat atau diproses sepenuhnya.")
        return

    try:
        # Membuat workbook dan worksheet baru di memori
        # Menggunakan nama file sementara yang lebih kecil kemungkinannya berkonflik
        temp_excel_file = 'temp_features_for_prediction.xlsx'
        workbook = xw.Workbook(temp_excel_file)
        worksheet = workbook.add_worksheet()

        # Mendefinisikan header untuk fitur GLCM
        glcm_feature_names = ['correlation', 'homogeneity', 'dissimilarity', 'contrast', 'energy', 'ASM']
        angles_names = ['0', '45', '90', '135']
        col_idx = 0
        for feat in glcm_feature_names:
            for angle_name in angles_names:
                worksheet.write(0, col_idx, f'{feat} {angle_name}')
                col_idx += 1

        # Mendefinisikan header untuk fitur HSV
        hsv_feature_names = ['hue', 'saturation', 'value']
        for feat in hsv_feature_names:
            worksheet.write(0, col_idx, feat)
            col_idx += 1

        # Mendefinisikan header untuk fitur Bentuk
        shape_feature_names = ['eccentricity', 'metric']
        for feat in shape_feature_names:
            worksheet.write(0, col_idx, feat)
            col_idx += 1

        # --- Ekstraksi Fitur GLCM ---
        distances = [5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 256
        symmetric = True
        normed = True

        # Memeriksa apakah grayscale valid untuk GLCM
        if grayscale.shape[0] < distances[0] * 2 or grayscale.shape[1] < distances[0] * 2:
            print("Peringatan: Gambar grayscale terlalu kecil untuk GLCM dengan jarak saat ini. Menggunakan nilai default.")
            glcm_props = [0.0] * (len(glcm_feature_names) * len(angles_names)) # Mengisi dengan nol atau menangani secara berbeda
        else:
            glcm = graycomatrix(grayscale, distances, angles, levels, symmetric=symmetric, normed=normed)
            glcm_props = [prop for name in glcm_feature_names for prop in graycoprops(glcm, name)[0]] # [0] karena distances adalah [5]

        col_idx = 0
        for item in glcm_props:
            worksheet.write(1, col_idx, item)
            col_idx += 1

        # --- Ekstraksi Fitur HSV ---
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Menggunakan img yang sudah dipotong
        mean_h = np.mean(hsv_img[:, :, 0])
        mean_s = np.mean(hsv_img[:, :, 1])
        mean_v = np.mean(hsv_img[:, :, 2])

        color_props = (mean_h, mean_s, mean_v)
        for item in color_props:
            worksheet.write(1, col_idx, item)
            col_idx += 1

        # --- Ekstraksi Fitur Bentuk ---
        # Memastikan mask adalah biner (0 atau 255) dan tidak kosong
        if mask.size == 0 or np.max(mask) == 0: # Memeriksa apakah mask sepenuhnya hitam atau kosong
            eccentricity = 0.0
            metric = 0.0
            print("Peringatan: Mask kosong atau semua hitam. Fitur bentuk diatur ke 0.")
        else:
            # Memastikan mask benar-benar biner sebelum pelabelan
            binary_mask = (mask > 0).astype(np.uint8) * 255
            label_image = label(binary_mask)
            props = regionprops(label_image)
            if props: # Memastikan regionprops menemukan region
                # Mendapatkan properti region terbesar (mengasumsikan itu daun)
                largest_region = max(props, key=lambda r: r.area)
                eccentricity = getattr(largest_region, 'eccentricity')
                area = getattr(largest_region, 'area')
                perimeter = getattr(largest_region, 'perimeter')
                metric = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
            else:
                eccentricity = 0.0
                metric = 0.0
                print("Peringatan: Tidak ada region berbeda yang ditemukan di mask. Fitur bentuk diatur ke 0.")

        worksheet.write(1, col_idx, eccentricity)
        col_idx += 1
        worksheet.write(1, col_idx, metric)

        workbook.close()

        # Membaca file Excel yang baru dibuat untuk prediksi
        hasil_ekstrak_df = pd.read_excel(temp_excel_file, sheet_name='Sheet1')

        if model is None:
            messagebox.showerror("Error Model", "Model klasifikasi belum dimuat. Pastikan 'model9.h5' ada.")
            return

        hasil_klasifikasi = np.argmax(model.predict(hasil_ekstrak_df), axis=-1)
        print(f"Hasil Klasifikasi (Internal): {hasil_klasifikasi}")

        # --- Mengevaluasi Akurasi Model (pada dataset terpisah) ---
        if enc is None:
             messagebox.showerror("Error Encoder", "LabelEncoder belum diinisialisasi. Pastikan 'datatesting.xlsx' ada dan kolom 'Keterangan' valid.")
             return

        try:
            datatesting = pd.read_excel("datatesting.xlsx")
            if 'Keterangan' in datatesting.columns:
                datatesting['Keterangan_Encoded'] = enc.transform(datatesting['Keterangan'].values)
                # Memastikan kolom fitur cocok dengan yang digunakan saat model dilatih
                # Praktik yang baik adalah menyelaraskan kolom jika disimpan dalam urutan tertentu
                # Untuk kode ini, mengasumsikan 'Keterangan' adalah satu-satunya kolom non-fitur di datatesting
                xtest_cols = [col for col in datatesting.columns if col not in ["Keterangan", "Keterangan_Encoded"]]
                xtest = datatesting[xtest_cols]
                ytest = datatesting['Keterangan_Encoded']

                # Melakukan penyelarasan jika diperlukan, meskipun Keras biasanya menanganinya jika bentuk input benar
                # Jika model Anda mengharapkan urutan fitur tertentu, pastikan xtest_df cocok.
                # Salah satu cara adalah memuat nama fitur dari data pelatihan Anda jika tersedia.
                # Untuk kesederhanaan di sini, mengasumsikan urutan fitur konsisten.

                loss, acc = model.evaluate(xtest, ytest, verbose=0) # verbose=0 untuk menekan output
                akurasi = round(acc, 4) * 100
                print(f"Akurasi Model (Internal): {akurasi:.2f}%")
            else:
                print("Peringatan: Kolom 'Keterangan' tidak ditemukan di datatesting.xlsx. Tidak dapat mengevaluasi akurasi.")
                akurasi = "N/A"
        except Exception as e:
            print(f"Error saat mengevaluasi akurasi model: {e}")
            akurasi = "N/A" # Mengatur akurasi ke Tidak Tersedia
            messagebox.showwarning("Peringatan Akurasi", f"Gagal menghitung akurasi model: {e}.\nPastikan 'datatesting.xlsx' valid dan sesuai format.")

        messagebox.showinfo("Informasi", "Ekstraksi ciri dan klasifikasi internal selesai!\nSilakan klik 'DETEKSI PENYAKIT' atau 'LIHAT AKURASI' untuk hasilnya.")

    except Exception as e:
        print(f"Error selama ekstraksi fitur atau klasifikasi: {e}")
        messagebox.showerror("Error Ekstraksi/Klasifikasi", f"Terjadi kesalahan saat ekstraksi ciri atau klasifikasi: {e}")
        hasil_klasifikasi = None # Mereset hasil pada error
        akurasi = None


def openDataset():
    # Fungsi ini memungkinkan pengguna untuk memilih direktori untuk dataset.
    # Ini tidak terintegrasi langsung ke alur klasifikasi untuk 'datatesting.xlsx'
    # tetapi dapat digunakan untuk tujuan lain (misalnya, memuat dataset pelatihan).
    global filedataset
    filedataset = filedialog.askdirectory(title="Pilih Direktori Dataset")
    if filedataset:
        messagebox.showinfo("Informasi", f"Direktori Dataset terpilih: {filedataset}")

def akurasiModel():
    if akurasi is not None and isinstance(akurasi, (int, float)):
        accuracy_text = f"{akurasi:.2f}%"
        output_accuracy_label.config(text=accuracy_text, fg='darkgreen')
    else:
        output_accuracy_label.config(text="Belum Tersedia", fg='red')
    if akurasi == "N/A":
        messagebox.showinfo("Informasi Akurasi", "Akurasi model belum tersedia atau terjadi kesalahan saat perhitungan.")
    else:
        messagebox.showinfo("Informasi Akurasi", f"Akurasi model adalah: {accuracy_text}")

def HasilKlasifikasi() :
    if hasil_klasifikasi is not None:
        disease_map = {
            0: "Antraknosa",
            1: "Bercak Daun (PSD)",
            2: "Hawar Daun",
            3: "Normal (Sehat)"
        }
        predicted_disease = disease_map.get(hasil_klasifikasi[0], "Tidak Diketahui")
        output_classification_label.config(text=predicted_disease, fg='blue')
        messagebox.showinfo("Hasil Klasifikasi", f"Penyakit Daun: {predicted_disease}")
    else:
        output_classification_label.config(text="Belum Klasifikasi", fg='red')
        messagebox.showwarning("Peringatan Klasifikasi", "Ekstraksi ciri belum dilakukan. Silakan klik 'EKSTRAK CIRI' terlebih dahulu.")

# --- Pengaturan Jendela Utama ---
window = tk.Tk()
window.configure(bg='lightgray') # Latar belakang yang sedikit lebih terang
window.geometry("1400x780") # Ukuran disesuaikan untuk tata letak grid yang lebih baik
window.title("KLASIFIKASI PENYAKIT DAUN TANAMAN CABAI") # Judul yang lebih spesifik

# --- Tata Letak Menggunakan Grid ---

# Mengonfigurasi bobot grid untuk perubahan ukuran responsif
window.grid_rowconfigure(0, weight=0) # Baris judul - tinggi tetap
window.grid_rowconfigure(1, weight=1) # Baris tampilan gambar - meluas
window.grid_rowconfigure(2, weight=0) # Baris tombol/output - tinggi tetap
window.grid_columnconfigure(0, weight=1) # Kolom utama - meluas

# Label Judul
title_label = tk.Label(window, text="KLASIFIKASI PENYAKIT DAUN TANAMAN CABAI",
                       font=("Cambria Bold", 18), fg="darkgreen", bg='lightgray')
title_label.grid(row=0, column=0, pady=15, columnspan=2) # columnspan untuk menengahkan

# --- Area Tampilan Gambar (Grid dari frame) ---
image_display_frame = tk.Frame(window, bg='white', bd=2, relief="groove")
image_display_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)

# Mengonfigurasi grid internal untuk frame tampilan gambar
for i in range(2): # 2 baris gambar
    image_display_frame.grid_rowconfigure(i, weight=1)
for i in range(3): # 3 kolom gambar
    image_display_frame.grid_columnconfigure(i, weight=1)

# Plot Fg1 (Normal)
frame_normal = tk.Frame(image_display_frame, bg='white', highlightbackground="black", highlightthickness="1")
frame_normal.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
fig1, ax1 = plt.subplots(figsize=(3, 3)) # Menyesuaikan figsize untuk sel grid
canvas1 = FigureCanvasTkAgg(fig1, master=frame_normal)
canvas1_widget = canvas1.get_tk_widget()
canvas1_widget.pack(fill="both", expand=True)
ax1.set_title("Normal")
ax1.axis('off') # Sembunyikan sumbu

# Plot Fg2 (Hue)
frame_hue = tk.Frame(image_display_frame, bg='white', highlightbackground="black", highlightthickness="1")
frame_hue.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
fig2, ax2 = plt.subplots(figsize=(3, 3))
canvas2 = FigureCanvasTkAgg(fig2, master=frame_hue)
canvas2_widget = canvas2.get_tk_widget()
canvas2_widget.pack(fill="both", expand=True)
ax2.set_title("Hue")
ax2.axis('off')

# Plot Fg3 (Saturation)
frame_saturation = tk.Frame(image_display_frame, bg='white', highlightbackground="black", highlightthickness="1")
frame_saturation.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
fig3, ax3 = plt.subplots(figsize=(3, 3))
canvas3 = FigureCanvasTkAgg(fig3, master=frame_saturation)
canvas3_widget = canvas3.get_tk_widget()
canvas3_widget.pack(fill="both", expand=True)
ax3.set_title("Saturation")
ax3.axis('off')

# Plot Fg4 (Value)
frame_value = tk.Frame(image_display_frame, bg='white', highlightbackground="black", highlightthickness="1")
frame_value.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
fig4, ax4 = plt.subplots(figsize=(3, 3))
canvas4 = FigureCanvasTkAgg(fig4, master=frame_value)
canvas4_widget = canvas4.get_tk_widget()
canvas4_widget.pack(fill="both", expand=True)
ax4.set_title("Value")
ax4.axis('off')

# Plot Fg5 (Grayscale)
frame_grayscale = tk.Frame(image_display_frame, bg='white', highlightbackground="black", highlightthickness="1")
frame_grayscale.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
fig5, ax5 = plt.subplots(figsize=(3, 3))
canvas5 = FigureCanvasTkAgg(fig5, master=frame_grayscale)
canvas5_widget = canvas5.get_tk_widget()
canvas5_widget.pack(fill="both", expand=True)
ax5.set_title("Grayscale")
ax5.axis('off')

# Plot Fg6 (Shape)
frame_shape = tk.Frame(image_display_frame, bg='white', highlightbackground="black", highlightthickness="1")
frame_shape.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
fig6, ax6 = plt.subplots(figsize=(3, 3))
canvas6 = FigureCanvasTkAgg(fig6, master=frame_shape)
canvas6_widget = canvas6.get_tk_widget()
canvas6_widget.pack(fill="both", expand=True)
ax6.set_title("Shape")
ax6.axis('off')

# --- Area Kontrol dan Output ---
control_output_frame = tk.Frame(window, bg='lightgray', bd=2, relief="groove")
control_output_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)

# Mengonfigurasi grid internal untuk control_output_frame
control_output_frame.grid_rowconfigure(0, weight=1) # Baris tombol
control_output_frame.grid_rowconfigure(1, weight=1) # Baris label output
control_output_frame.grid_columnconfigure([0,1,2,3,4,5,6], weight=1) # Membuat tombol meluas secara merata

# Tombol untuk pemrosesan gambar dan ekstraksi fitur
button_frame = tk.Frame(control_output_frame, bg='lightgray')
button_frame.grid(row=0, column=0, columnspan=7, pady=5, padx=5, sticky="ew") # Meliputi semua kolom untuk tombol
button_frame.grid_columnconfigure([0,1,2,3,4,5,6], weight=1) # Membuat tombol meluas secara merata

tk.Button(button_frame, text="OPEN IMAGE", command=openImage, height=2, width=15,
          font=('Cambria', 10, 'bold'), bg='#4CAF50', fg='white', relief="raised").grid(row=0, column=0, padx=3, pady=3, sticky="ew")
tk.Button(button_frame, text="HUE", command=Hue, height=2, width=15,
          font=('Cambria', 10, 'bold'), bg='#2196F3', fg='white', relief="raised").grid(row=0, column=1, padx=3, pady=3, sticky="ew")
tk.Button(button_frame, text="SATURATION", command=Saturation, height=2, width=15,
          font=('Cambria', 10, 'bold'), bg='#2196F3', fg='white', relief="raised").grid(row=0, column=2, padx=3, pady=3, sticky="ew")
tk.Button(button_frame, text="VALUE", command=Value, height=2, width=15,
          font=('Cambria', 10, 'bold'), bg='#2196F3', fg='white', relief="raised").grid(row=0, column=3, padx=3, pady=3, sticky="ew")
tk.Button(button_frame, text="GRAYSCALE", command=Grayscales, height=2, width=15,
          font=('Cambria', 10, 'bold'), bg='#2196F3', fg='white', relief="raised").grid(row=0, column=4, padx=3, pady=3, sticky="ew")
tk.Button(button_frame, text="SHAPE", command=Shape, height=2, width=15,
          font=('Cambria', 10, 'bold'), bg='#2196F3', fg='white', relief="raised").grid(row=0, column=5, padx=3, pady=3, sticky="ew")
tk.Button(button_frame, text="EKSTRAK CIRI", command=ekstrak_ciri, height=2, width=20,
          font=('Cambria', 10, 'bold'), bg='#FF9800', fg='white', relief="raised").grid(row=0, column=6, padx=3, pady=3, sticky="ew")

# Bagian Output Klasifikasi dan Akurasi
output_section_frame = tk.Frame(control_output_frame, bg='lightgray')
output_section_frame.grid(row=1, column=0, columnspan=7, pady=10, padx=5, sticky="ew")
output_section_frame.grid_columnconfigure([0,1,2,3], weight=1) # Untuk label output

# Output Klasifikasi
tk.Label(output_section_frame, text="HASIL KLASIFIKASI:", font=('Cambria Bold', 10), bg='lightgray', fg='darkblue').grid(row=0, column=0, padx=5, pady=2, sticky="w")
output_classification_label = tk.Label(output_section_frame, text="Belum Klasifikasi", font=('Cambria Bold', 12),
                                       bg='white', highlightbackground="black", highlightthickness="1",
                                       width=25, anchor="center")
output_classification_label.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
tk.Button(output_section_frame, text="DETEKSI PENYAKIT", command=HasilKlasifikasi, height=1,
          font=('Cambria', 9, 'bold'), bg='#00BCD4', fg='white', relief="raised").grid(row=2, column=0, padx=5, pady=5, sticky="ew")

# Spacer untuk pemisahan visual
tk.Frame(output_section_frame, width=20, bg='lightgray').grid(row=0, column=1, rowspan=3)


# Output Akurasi
tk.Label(output_section_frame, text="AKURASI MODEL:", font=('Cambria Bold', 10), bg='lightgray', fg='darkblue').grid(row=0, column=2, padx=5, pady=2, sticky="w")
output_accuracy_label = tk.Label(output_section_frame, text="N/A", font=('Cambria Bold', 12),
                                 bg='white', highlightbackground="black", highlightthickness="1",
                                 width=15, anchor="center")
output_accuracy_label.grid(row=1, column=2, padx=5, pady=2, sticky="ew")
tk.Button(output_section_frame, text="LIHAT AKURASI", command=akurasiModel, height=1,
          font=('Cambria', 9, 'bold'), bg='#9C27B0', fg='white', relief="raised").grid(row=2, column=2, padx=5, pady=5, sticky="ew")


# Panggilan awal untuk memastikan plot kosong dan sumbu mati
clear_all_plots()
ax1.clear()
ax1.set_title("Normal (Diproses)")
ax1.axis('off')
canvas1.draw()


window.mainloop()
