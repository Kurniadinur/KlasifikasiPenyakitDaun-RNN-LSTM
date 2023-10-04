import tkinter as tk
from tkinter import filedialog
import numpy as np 
import pandas as pd
import cv2
import xlsxwriter as xw
from sklearn.preprocessing import LabelEncoder
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score
from tkinter import *
from rembg import remove
from skimage.feature import graycomatrix,graycoprops
from skimage.measure import label, regionprops
from PIL import Image

window = tk.Tk()


window.configure (bg='grey')
window.geometry("1320x730")
window.title ("KLASIFIKASI PENYAKIT DAUN")

def openImage():
    global fileImage
    global img
    global img_HSV
    global mask
    global grayscale
    fileImage = filedialog.askopenfilename()
    input = Image.open(fileImage)
    a,b = input.size
    rimg = remove(input)
    image = rimg.resize((int(b/2),int(a/2)))
    img = np.array(image)
    change  = img[:,:,3]==0
    img[change]=[255,255,255,255]
    tmp = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask.copy(), None, iterations=10)
    mask = cv2.erode(mask.copy(), None, iterations=10)
    b, g, r    = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    selected = max(contours,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(selected)
    mask = mask[y:y+h,x:x+w]
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    #grayscale gambar
    ax1.imshow(img)
    canvas1.draw()
    ax2.imshow(H)
    canvas2.draw()
    ax3.imshow(S)
    canvas3.draw()
    ax4.imshow(V)
    canvas4.draw()
    ax5.imshow(grayscale, cmap='gray')
    canvas5.draw()
    ax6.imshow(mask, cmap='gray')
    canvas6.draw()


def ekstrak_ciri():
    global hasil_klasifikasi
    global akurasi
    # img = cv2.imread(fileImage)

    data_baru = xw.Workbook('test.xlsx')
    tambah = data_baru.add_worksheet()

    kolom = 0
    #kolom feature GLCM
    glcm_feature = ['correlation','homogeneity','dissimilarity', 'contrast', 'energy', 'ASM']
    sudut = ['0','45','90','135']
    for i in glcm_feature :
        for j in sudut :
            tambah.write (0,kolom,i+' '+j)
            kolom+=1
    #kolom feature HSV
    hsv_feature = ['hue','saturation', 'value']
    for i in hsv_feature:
        tambah.write(0,kolom,i)
        kolom+=1
    #kolom feature Shape
    shape_feature = ['accentricity', 'metric']
    for i in shape_feature :
        tambah.write(0,kolom,i)
        kolom+=1
    
    kolom = 0
    #GLCM Feature
    distances = [5]
    angles = [0,np.pi/4,np.pi/2,3*np.pi/4]
    levels = 256
    symetric = True
    normed = True

    glcm = graycomatrix (grayscale, distances, angles, levels, symetric, normed)
    glcm_props = [propery for name in glcm_feature for propery in graycoprops(glcm,name)[0]]
    for item in glcm_props :
        tambah.write (1,kolom,item)
        kolom+=1

    #Feature HSV
    hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    mean_h = np.mean(H)
    mean_s = np.mean(S)
    mean_v = np.mean(V)

    color_props = (mean_h,mean_s,mean_v)
    for item in color_props:
        tambah.write(1,kolom,item)
        kolom+=1   

    #feature Shape
    label_image     = label(mask)
    props           = regionprops(label_image)
    eccentricity    = getattr(props[0], 'eccentricity')
    area            = getattr(props[0], 'area')
    perimeter       = getattr(props[0], 'perimeter')
    if perimeter == 0:
        metric = 0
    else:
        metric = (4*np.pi*area)/(perimeter*perimeter)
    tambah.write(1,kolom,eccentricity)
    kolom+=1
    tambah.write(1,kolom,metric)
    kolom+=1

    data_baru.close()
    hasil_ekstrak = pd.read_excel('test.xlsx',sheet_name='Sheet1')
    model = load_model("model5.h5")
    hasil_klasifikasi = np.argmax(model.predict(hasil_ekstrak),axis=-1)
    print(hasil_klasifikasi)


    datatesting = pd.read_excel("datatesting.xlsx")
    enc = LabelEncoder()
    datatesting['Keterangan'] = enc.fit_transform (datatesting['Keterangan'].values)
    xtest = datatesting.drop(columns="Keterangan")
    ytest = datatesting['Keterangan']
    loss, acc = model.evaluate(xtest,ytest, verbose =2)
    akurasi = round(acc,4)*100
    print (round(acc,4)*100,'%')
    
def openDataset():
    global filedataset
    filedataset = filedialog.askdirectory()

def akurasiModel():
    Ouput = tk.Label(frame10, font=('Cambria Bold',11) ,text=(str(akurasi),'%'), background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)
    
def HasilKlasifikasi() :
    if hasil_klasifikasi[0] == 0:
        Ouput = tk.Label(frame7, font=('Cambria Bold',11) ,text="Antranoksa", background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)
    if hasil_klasifikasi[0] == 1:
        Ouput = tk.Label(frame7, font=('Cambria Bold',11) ,text="PSD", background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)
    if hasil_klasifikasi[0] == 2:
        Ouput = tk.Label(frame7, font=('Cambria Bold',11) ,text="Hawar", background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)
    if hasil_klasifikasi[0] == 3:
        Ouput = tk.Label(frame7, font=('Cambria Bold',11) ,text="Normal", background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)
#
# label = tk.Label (window,text="KLASIFIKASI PENYAKIT DAUN PADA DAUN BLALA", font=("Cambria Bold",12),fg="black")
# label.place( x=380, y=2)

#Frame gambar Normal
frame2 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame2.place(x=50, y=20)
#Plot Fg1
fig1, ax1 = plt.subplots()
fig1.set_size_inches(w=2.8,h=2.8)
canvas1 = FigureCanvasTkAgg (fig1, master=frame2)
canvas1.get_tk_widget().pack()
ax1.set_title ("Normal")

#frame plot Hue
frame3 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame3.place(x=350, y=20)
#Plot Fg2
fig2, ax2 = plt.subplots()
fig2.set_size_inches(w=2.8,h=2.8)
canvas2 = FigureCanvasTkAgg (fig2, master=frame3)
canvas2.get_tk_widget().pack()
ax2.set_title ("Hue")

#frame plot Saturation
frame4 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame4.place(x=650, y=20)
#Plot Fg2
fig3, ax3 = plt.subplots()
fig3.set_size_inches(w=2.8,h=2.8)
canvas3 = FigureCanvasTkAgg (fig3, master=frame4)
canvas3.get_tk_widget().pack()
ax3.set_title ("Saturation")

#frame plot Value
frame5 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame5.place(x=950, y=20)
#Plot Fg2
fig4, ax4 = plt.subplots()
fig4.set_size_inches(w=2.8,h=2.8)
canvas4 = FigureCanvasTkAgg (fig4, master=frame5)
canvas4.get_tk_widget().pack()  
ax4.set_title ("Value")

#frame plot Grayscale
frame8 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame8.place(x=350, y=320)
#Plot Fg5
fig5, ax5 = plt.subplots()
fig5.set_size_inches(w=2.8,h=2.8)
canvas5 = FigureCanvasTkAgg (fig5, master=frame8)
canvas5.get_tk_widget().pack()  
ax5.set_title ("Grayscale")

#frame plot Shape
frame9 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame9.place(x=650, y=320)
#Plot Fg5
fig6, ax6 = plt.subplots()
fig6.set_size_inches(w=2.8,h=2.8)
canvas6 = FigureCanvasTkAgg (fig6, master=frame9)
canvas6.get_tk_widget().pack()  
ax6.set_title ("Shape")

#Frame Bawahhh
frame6 = tk.Frame (window, width=600, height=100, highlightbackground="black", highlightthickness="1",background='white')
frame6.place(x=50, y=620)
#Frame Hasil
frame7 = tk.Frame (window, width=305, height=100, highlightbackground="black", highlightthickness="1",background='white')
frame7.place(x=980, y=620)
#Frame Akurasi
frame10 = tk.Frame (window, width=305, height=100, highlightbackground="black", highlightthickness="1",background='white')
frame10.place(x=662, y=620)

#Tombol Open dan Esktrak
open_image = tk.Button (frame6, text="OPEN IMAGE", command = openImage, height=2,width=20).place(x=140,y=30)
ekstrak_citra= tk.Button (frame6, text="EKSTRAK CIRI", command= ekstrak_ciri, height=2,width=20).place (x=300, y=30)
#Tombol Hasil Hasil
tombol_klasifikasi = tk.Button(frame7, text="DETEKSI PENYAKIT DAUN", command= HasilKlasifikasi,height=1,width=25).place(x=62, y=15)
Ouput = tk.Label(frame7, font=('Cambria Bold',11) ,text="", background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)
#Tombol Akurasi
tombol_akurasi = tk.Button(frame10, text="AKURASI MODEL", command= akurasiModel,height=1,width=25).place(x=62, y=15)
Ouput = tk.Label(frame10, font=('Cambria Bold',11) ,text="", background='white', highlightbackground="black", highlightthickness="2", width= 22).place(x=50,y=50)

window.mainloop()