import numpy as np
import sys
import cv2
import math
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QSlider
from PIL import ImageQt
import pandas
from openpyxl.workbook import Workbook



# A1

class ShowImage(QMainWindow):

    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('untitled.ui', self)
        self.Image = None
        self.pushButton.clicked.connect(self.sip)  # A2
        self.pushButton_2.clicked.connect(self.save)
        self.actionGrayScale.triggered.connect(self.grayClicked)  # A3
        self.actionOperasi_Pencerahan.triggered.connect(self.brigtness)  # A4
        self.actionsimple_contrast.triggered.connect(self.contrast)  # A5
        self.horizontalSlider.valueChanged[int].connect(self.contrast)
        self.horizontalSlider_2.valueChanged[int].connect(self.brigtness)
        self.actionContrast_Stretching.triggered.connect(self.Stretching)  # A6
        self.actionNegative_Image.triggered.connect(self.Negative)  # A7
        self.actionBiner_Image.triggered.connect(self.Biner)  # A8
        self.actionGrayScale_2.triggered.connect(self.histgrayscale)  # A9
        self.actionRGB_Histogram_2.triggered.connect(self.RGBHISTO)  # A10
        self.actionEquals_2.triggered.connect(self.Equal)  # A11
        self.actionTranslasi_2.triggered.connect(self.Trans)  # B1
        self.actionTranspose_2.triggered.connect(self.rotasiTranspose)
        self.action180_Degree.triggered.connect(self.seratusdegree)
        self.action90_Degree.triggered.connect(self.sembilandegree)
        self.action45_Degree.triggered.connect(self.empatlima)
        self.action_45_Degree.triggered.connect(self.minusempat)
        self.action_90_Degree.triggered.connect(self.minussembilan)
        self.action2x_2.triggered.connect(self.zoomin)
        self.action3x_2.triggered.connect(self.zoom3x)
        self.action4x_2.triggered.connect(self.zoom4x)
        self.action1_3.triggered.connect(self.zoomout)
        self.action1_5.triggered.connect(self.zoom14)
        self.action3_5.triggered.connect(self.zoom34)
        self.actionDimensi_2.triggered.connect(self.dimensi)
        self.actionCrop_Image.triggered.connect(self.crop)
        self.actionadd.triggered.connect(self.aritmatika)
        self.actionsub.triggered.connect(self.aritmatikasub)
        self.actionmul.triggered.connect(self.aritmatikamul)
        self.actiondiv.triggered.connect(self.aritmatikadiv)
        self.actionor.triggered.connect(self.BOOLEAN)
        self.actionand.triggered.connect(self.BOOLEANand)
        self.actionxor.triggered.connect(self.BOOLEANxor)
        self.actionModul.triggered.connect(self.filteringModul)
        self.actionTugas_A.triggered.connect(self.filteringTugasA)
        self.actionTugas_B.triggered.connect(self.filteringTugasB)
        self.actionKernel_1.triggered.connect(self.smoothing1)
        self.actionKernel_2.triggered.connect(self.smoothing2)
        self.actionGaussian.triggered.connect(self.gaussian)
        self.actionKernel_3.triggered.connect(self.Imagesharp1)
        self.actionKernel_4.triggered.connect(self.Imagesharp2)
        self.actionKernel_5.triggered.connect(self.Imagesharp3)
        self.actionKernel_6.triggered.connect(self.Imagesharp4)
        self.actionKernel_7.triggered.connect(self.Imagesharp5)
        self.actionKernel_8.triggered.connect(self.Imagesharp6)
        self.actionKernel_tugas.triggered.connect(self.Imagesharptugas)
        self.actionMedian.triggered.connect(self.medianImage)
        self.actionMax_Filter.triggered.connect(self.MaxFilter)
        self.actionMin_Filter.triggered.connect(self.MinFilter)
        self.actionDFT_Smooting_Image_Tepi.triggered.connect(self.DFTTepi)
        self.actionSobel.triggered.connect(self.Sobel)
        self.actionPewitt.triggered.connect(self.Prewitt)
        self.actionRoberts.triggered.connect(self.Robets)
        self.actionCanny.triggered.connect(self.canny)
        self.actionDFT_Smooting_Image.triggered.connect(self.DFT)
        self.actionoke.triggered.connect(self.Project)
        
    # A2
    def sip(self):
        img, _ = QFileDialog.getOpenFileName()
        if img == "":
            return
        self.Image = cv2.imread(img)
        pixmap = QPixmap(img)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)
        self.displayImage()


    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(None, 'Save File', "", "Image File (*.jpg *.png *.xlsx)")

        if filePath == "":
            return

        image = ImageQt.fromqpixmap(self.label_10.pixmap())

        image.save(filePath)

    # A3
    def grayClicked(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] + 0.587 * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gray") + ".xlsx")

        self.displayImage(2)

    # A4
    def brigtness(self,value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = value
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Brightness") + ".xlsx")

    # A5
    def contrast(self,value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = value
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = a * contrast

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Contrast") + ".xlsx")

    # A6
    def Stretching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Stretching") + ".xlsx")

    # A7
    def Negative(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = math.ceil(255 - a)

                self.Image.itemset((i, j), b)
        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Negative") + ".xlsx")

    # A8
    def Biner(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = self.Image.item(i, j)
                T = 190
                if a == T:
                    b = 0
                elif a < T:
                    b = 1
                elif a > T:
                    b = 255

                self.Image.itemset((i, j), b)

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Biner") + ".xlsx")

    # A9
    def histgrayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        print(H, " ", W)

        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    # A10
    def RGBHISTO(self):  # fungsi RGBHISTO dengan parameter self
        color = ('b', 'g', 'r')  # variabel color yang berisikan 'b', 'g', 'r'
        for i, col in enumerate(color):  # enumerate sendiri berfungsi untuk
            # menambahkan penghitung (indeks) ke objek
            # dan mengembalikannya.
            # looping dengan variable i dan col pada variabel color yang sudah di enumerate
            histo = cv2.calcHist([self.Image], [i], None, [256],
                                 [0, 256])  # variabel histo yang berisikan method calcHist yang berfungsi
            # mencari dan menghitung histogram di suatu lokasi gambar tertentu.
            # gambar disini menggunakan self.Image dan rentang untuk histogramnya 0-256

        plt.plot(histo, color=col)  # plt.plot berfungsi untuk membuat kurva
        # dengan data dari variable histo dan dengan
        # warna pada variabel col pada looping yaitu rgb
        plt.xlim([0, 256])  # plt.xlim berfungsi untuk menyesuaikan batas sumbu x pada kurva
        # disini panjang sumbu x adalaah 0 sampai 256
        plt.show()  # plt.show berfungsi menampilkan plot yang telah dibuat

    # A11
    def Equal(self):  # fungsi Equal dengan parameter self
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])  # variable hist dan bins yang berisi
        # method numpy histogram dengan parameter
        # self.Image.flatten dengan rentang 0 sampai 255
        cdf = hist.cumsum()  # Fungsi cumsum yaitu menghitung jumlah kumulatif
        # nilai dalam larik dan menghasilkan larik keluaran baru.
        cdf_normalized = cdf * hist.max() / cdf.max()  # variabel normalized yang
        # berisi variabel cdf dikali
        # dengan variabel (hist.max() / cdf.max())
        cdf_m = np.ma.masked_equal(cdf,
                                   0)  # fungsi dari masked_equal yaitu mendapatkan nilai dari mask dalam array 'cdf'
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (
                    cdf_m.max() - cdf_m.min())  # rumus min max yang bertujuan untuk mendapatkan hasil stretching
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")  # variabel cdf
        self.Image = cdf[self.Image]  #
        self.displayImage(2)  # menampilkan fungsi displayImage dengan parameter 2

        plt.plot(cdf_normalized, color='b')  # membuat plot dari variabel cdf_normalized dengan warna biru
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')  # membuat histogram dari self.Image.flatten
        # dengan rentang 0-256 dengan warna merah
        plt.xlim([0, 256])  # membuat rentang kurva sumbu x dari 0 sampai 256
        plt.legend(("cdf", "histogram"), loc="upper left")  # membuat keterangan/legenda dari kurva yang
        # berisikan 'cdf' dan 'histogram' yang
        # berlokasi di atas kiri
        plt.show()  # plt.show berfungsi menampilkan plot yang telah dibuat


    def Trans(self):
        h, w = self.Image.shape[:2]
        height = 2
        widht = 5
        quarter_h, quarter_w = h / height, w / widht
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(2)


    def seratusdegree(self):
        self.rotasi(180)

    def sembilandegree(self):
        self.rotasi(90)

    def empatlima(self):
        self.rotasi(45)

    def minusempat(self):
        self.rotasi(-45)

    def minussembilan(self):
        self.rotasi(-90)

    def rotasi(self, value):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), value, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_Image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_Image

        self.displayImage(2)


    def rotasiTranspose(self):
        rot_Image = cv2.transpose(self.Image)
        self.Image = rot_Image
        self.displayImage(2)


    def zoomin(self):
        zoomin = 2
        resize_img = cv2.resize(self.Image, None, fx=zoomin, fy=zoomin, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def zoom3x(self):
        zoomin = 3
        resize_img = cv2.resize(self.Image, None, fx=zoomin, fy=zoomin, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def zoom4x(self):
        zoomin = 4
        resize_img = cv2.resize(self.Image, None, fx=zoomin, fy=zoomin, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def zoomout(self):
        zoomout = 1 / 2
        resize_img = cv2.resize(self.Image, None, fx=zoomout, fy=zoomout)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def zoom14(self):
        zoomout = 1 / 4
        resize_img = cv2.resize(self.Image, None, fx=zoomout, fy=zoomout)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def zoom34(self):
        zoomout = 3 / 4
        resize_img = cv2.resize(self.Image, None, fx=zoomout, fy=zoomout)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def dimensi(self):
        resize_img = cv2.resize(self.Image, (900, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def crop(self):
        H, W = self.Image.shape[:2]
        row_awal = 300
        col_awal = 400
        row_akhir = H
        col_akhir = W
        resize_img = self.Image[row_awal:row_akhir, col_awal:col_akhir]
        cv2.imshow("Zoom In", resize_img)
        cv2.waitKey()


    def aritmatika(self):

        img1 = cv2.imread("wallpaper.jpg", 0)
        img2 = cv2.imread("wallpaper1.jpg", 0)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        add_img = img1 + img2
        self.Image = add_img
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Tambah") + ".xlsx")
        self.displayImage(2)

    def aritmatikasub(self):

        img1 = cv2.imread("wallpaper.jpg", 0)
        img2 = cv2.imread("wallpaper1.jpg", 0)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        subtract = img1 - img2
        self.Image = subtract
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Kurang") + ".xlsx")
        self.displayImage(2)

    def aritmatikamul(self):

        img1 = cv2.imread("wallpaper.jpg", 0)
        img2 = cv2.imread("wallpaper1.jpg", 0)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        multiple = img1 * img2
        self.Image = multiple
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Perkalian") + ".xlsx")
        self.displayImage(2)

    def aritmatikadiv(self):

        img1 = cv2.imread("wallpaper.jpg", 0)
        img2 = cv2.imread("wallpaper1.jpg", 0)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        division = img1 / img2
        self.Image = division
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Pembagian") + ".xlsx")
        self.displayImage(2)


    def BOOLEAN(self):
        img1 = cv2.imread("wallpaper.jpg", 1)
        img2 = cv2.imread("wallpaper1.jpg", 1)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_or = cv2.bitwise_or(img1, img2)
        self.Image = op_or
        #df = pandas.DataFrame(op_or)
        #df.to_excel(str("Boolean OR") + ".xlsx")
        self.displayImage(2)

    def BOOLEANand(self):
        img1 = cv2.imread("wallpaper.jpg", 1)
        img2 = cv2.imread("wallpaper1.jpg", 1)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_and = cv2.bitwise_and(img1, img2)
        self.Image = op_and
        #df = pandas.DataFrame(self.Image)
        #df.to_excel(str("Boolean AND") + ".xlsx")
        self.displayImage(2)

    def BOOLEANxor(self):
        img1 = cv2.imread("wallpaper.jpg", 1)
        img2 = cv2.imread("wallpaper1.jpg", 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        cv2.imshow("Ori 1", img1)
        cv2.imshow("ori 2", img2)
        op_xor = cv2.bitwise_xor(img1, img2)
        self.Image = op_xor
        #df = pandas.DataFrame(self.Image)
        #df.to_excel(str("Boolean XOR") + ".xlsx")
        self.displayImage(2)

    def conv(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]

        F_height = F.shape[0]
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum

        return out

    def conv2(X, F):
        X_Height = X.shape[0]
        X_Width = X.shape[1]
        F_Height = F.shape[0]
        F_Width = F.shape[1]
        H = 0
        W = 0
        batas = (F_Height) // 2
        out = np.zeros((X_Height, X_Width))
        for i in np.arange(H, X_Height - batas):
            for j in np.arange(W, X_Width - batas):
                sum = 0
                for k in np.arange(H, F_Height):
                    for l in np.arange(W, F_Width):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def filteringModul(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Filtering") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def filteringTugasA(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Filtering Tugas A") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def filteringTugasB(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[6, 0, -6],
             [6, 1, -6],
             [6, 0, -6]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Filtering Tugas B") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def smoothing1(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Smoothing") + ".xlsx")
        #self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def smoothing2(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[1 / 4, 1 / 4],
             [1 / 4, 1 / 4]])

        filt = self.conv(img, conv)
        awa = filt
        awb = filt
        H, W = awa.shape[:2]
        imge = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                b = awb.item(i, j)
                imge.itemset((i, j), b)

        df = pandas.DataFrame(imge)
        df.to_excel(str("Smoothing 2") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def gaussian(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = (1 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian") + ".xlsx")
        #self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()
    def Imagesharp1(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel 1") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def Imagesharp2(self):
        load, _ = QFileDialog.getOpenFileName()
        img = cv2.imread(load, cv2.IMREAD_GRAYSCALE)
        conv = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel 2") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def Imagesharp3(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel 3") + ".xlsx")
        self.displayImage(2)

    def Imagesharp4(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[1, -2, 1],
             [-2, 5, -2],
             [1, -2, 1]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel 4") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def Imagesharp5(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel 5") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def Imagesharp6(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel 6") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def Imagesharptugas(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = (1.0 / 16) * np.array([[0, 0, -1, 0, 0],
                                      [0, -1, -2, -1, 0],
                                      [-1, -2, 16, -2, -1],
                                      [0, -1, -2, -1, 0],
                                      [0, 0, -1, 0, 0]])

        filt = self.conv(img, conv)
        self.Image = filt
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Gaussian Kernel Laplace") + ".xlsx")
        # self.displayImage(2)
        plt.figure(figsize=(5, 5))
        plt.imshow(filt, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()
        cv2.waitKey()

    def medianImage(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]


        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)
                self.Image = img_out

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Median Filtering") + ".xlsx")


    def MaxFilter(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                img_out.itemset((i, j), b)
                self.Image = img_out

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Max Filtering") + ".xlsx")

    def MinFilter(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                b = min
                img_out.itemset((i, j), b)
                self.Image = img_out

        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Min Filtering") + ".xlsx")

    def DFT(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrume = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r

        mask[mask_area] = 1
        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrume, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse fourier')
        plt.show()

    def DFTTepi(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrume = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r

        mask[mask_area] = 0
        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrume, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse fourier')
        plt.show()

    def Sobel(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = self.conv(img, Sx)
        img_y = self.conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        self.Image = img_out
        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Sobel") + ".xlsx")
        print(self.Image)
        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Prewitt(self):

        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
        img_x = self.conv(img, Sx)
        img_y = self.conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        img = img_out
        self.Image = img
        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Prewitt") + ".xlsx")
        print(self.Image)
        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Robets(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[1, 0],
                       [0, -1]])
        Sy = np.array([[0, 1],
                       [-1, 0]])
        img_x = self.conv2(img, Sx)
        img_y = self.conv2(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        self.Image = img_out
        self.displayImage(2)
        df = pandas.DataFrame(self.Image)
        df.to_excel(str("Robets") + ".xlsx")
        print(self.Image)
        print(img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def canny(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        conv = (1 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])

        out_img = self.conv(img, conv)
        out_img = out_img.astype("uint8")
        df = pandas.DataFrame(out_img)
        df.to_excel(str("Noise reduction") + ".xlsx")
        cv2.imshow("Noise reduction",out_img)
        print(out_img)
        #finding gradien
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = self.conv(out_img, Sx)
        img_y = self.conv(out_img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        cv2.imshow("finding Gradien",img_out)
        theta = np.arctan2(img_y, img_x)
        df = pandas.DataFrame(img_out)
        df.to_excel(str("finding Gradien") + ".xlsx")
        print(img_out)
        #non maximum
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        H, W = img.shape[:2]
        Z = np.zeros((H,W), dtype=np.int32 )
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                        # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                        # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                        # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]
                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i,j] = img_out[i, j]
                    else:
                        Z[i,j] = 0
                except IndexError as e:
                    pass
        img_N = Z.astype("uint8")
        cv2.imshow("Non Maximum Suppression" , img_N)
        df = pandas.DataFrame(img_N)
        df.to_excel(str("Non Maximum Suppression") + ".xlsx")
        print(img_N)
        #hysteresis treshold part 1
        weak = 100
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak) :
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        cv2.imshow("hysteresis part 1", img_H1)
        df = pandas.DataFrame(img_H1)
        df.to_excel(str("hysteresis part 1") + ".xlsx")
        print(img_H1)
        #hysteresis treshold part 2
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or (
                                img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or (
                                img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or (
                                img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")
        print(img_H2)
        cv2.imshow("hysteresis part 2", img_H2)
        df = pandas.DataFrame(img_H2)
        df.to_excel(str("hysteresis part 2") + ".xlsx")

    def Project(self):
        img = self.Image
        H, W = img.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 2], 0, 255)

        conv = (1 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])
        filt = self.conv(gray, conv)

        try:
            imga = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)

        except:
            pass
        imga = filt

        H, W = filt.shape[:2]
        T = 100
        for i in range(H):
            for j in range(W):
                a = filt.item(i, j)
                b = filt.item(i, j)

                if a == T:
                    b = 0
                elif a < T:
                    b = 0
                elif a > T:
                    b = 255
                filt.itemset((i, j), b)

        # filt = self.conv(imga, conv)

        H, W = filt.shape[:2]

        for i in range(H):
            for j in range(W):
                a = filt.item(i, j)
                b = math.ceil(255 - a)

                filt.itemset((i, j), b)
        imgh = img
        awb = filt
        filt = filt.astype("uint8")
        self.Image = filt
        self.displayImage(4)
        df = pandas.DataFrame(filt)
        df.to_excel(str("Citra Yang sudah Diperbaiki Kualitas citranya") + ".xlsx")

        H, W = filt.shape[:2]
        imge = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                b = awb.item(i, j)
                imge.itemset((i, j), b)
        dst = cv2.inpaint(img, filt, 3, cv2.INPAINT_TELEA)

        self.Image = dst
        self.displayImage(2)
        df = pandas.DataFrame(imge)
        df.to_excel(str("Menghaluskan Object") + ".xlsx")
        awa = filt
        hierarchy, _ = cv2.findContours(awa, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in hierarchy:
            epsilon = 0.009 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            x, y = approx[0][0]

            if len(approx) == 3:
                cv2.putText(imgh, "", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif len(approx) == 4:
                cv2.putText(imgh, "", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif len(approx) == 5:
                cv2.putText(imgh, "", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif 6 < len(approx) < 15:
                cv2.putText(imgh, "", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            else:
                cv2.putText(imgh, "", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)

        self.Image = imgh
        self.displayImage(3)
        H, W = awa.shape[:2]
        imge = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                b = awb.item(i, j)
                imge.itemset((i, j), b)

        df = pandas.DataFrame(imge)
        df.to_excel(str("detect") + ".xlsx")

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        elif windows == 2:
            self.label_10.setPixmap(QPixmap.fromImage(img))
            self.label_10.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_10.setScaledContents(True)
        elif windows == 3:
            self.label_12.setPixmap(QPixmap.fromImage(img))
            self.label_12.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_12.setScaledContents(True)
        elif windows == 4:
            self.label_14.setPixmap(QPixmap.fromImage(img))
            self.label_14.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_14.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('pertemuan 3')
window.show()
sys.exit(app.exec_())