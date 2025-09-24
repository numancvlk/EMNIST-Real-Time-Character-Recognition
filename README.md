# EMNIST - Real Time Character Recognition

# [TR]
## Projenin Amacı
Bu proje, **EMNIST** el yazısı rakam ve harf verisini kullanarak bir CNN modeli eğitmek ve ardından gerçek zamanlı kamera görüntüsü üzerinden rakam ve harf tahmini yapmak amacıyla geliştirilmiştir.

## 💻 Kullanılan Teknolojiler
- Python 3.11.8
- PyTorch: Model oluşturma, eğitim ve tahmin için.
- Torchvision: MNIST dataset ve transform işlemleri için.
- OpenCV: Kamera görüntüsü yakalamak için
- Matplotlib: Eğitim verilerini ve tahmin sonuçlarını görselleştirmek için.
- TQDM: Eğitim sürecinde ilerleme çubuğu göstermek için.
- Timeit: Eğitim süresini ölçmek için.

## ⚙️ Kurulum
GEREKLİ KÜTÜPHANELERİ KURUN
```bash
pip install torch torchvision matplotlib opencv-python tqdm
```

## 🚀 Çalıştırma
1. Önce veriler için **dataset.py** dosyasını çalıştırın.
2. Modeli oluşturmak için **model.py** dosyasını çalıştırın.
3. Modelin ihtiyaç duyduğu fonksiyonlar için **helpers.py** dosyasını çalıştırın.
4. Modeli eğitmek için **train.py** dosyasını çalıştırın.
5. Modelin tahminleri için **predict.py** dosyasını çalıştırın.
6. Modelin kamera üzerinden tahminler yapmasını sağlamak için **realTimePredict.py** dosyasını çalıştırın.

## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.

# [EN]
## Project Objective
This project is developed to train a CNN model using EMNIST handwritten digit and letter data and then make digit and letter predictions in real-time through camera input.

## 💻 Technologies Used
- Python 3.11.8
- PyTorch: For model creation, training, and prediction.
- Torchvision: For MNIST dataset and transform operations.
- OpenCV: To capture camera input.
- Matplotlib: To visualize training data and prediction results.
- TQDM: To show a progress bar during training.
- Timeit: To measure training time.

## ⚙️ Installation

INSTALL REQUIRED LIBRARIES
```bash
pip install torch torchvision matplotlib opencv-python tqdm
```

## 🚀 How to Run

1. First, run the **dataset.py** file for the data.
2. Run **model.py** to create the model.
3. Run **helpers.py** for the functions required by the model.
4. Run **train.py** to train the model.
5. Run **predict.py** to make predictions using the model.
6. Run **realTimePredict.py** to enable the model to make predictions through the camera.

THIS PROJECT DOES NOT CONTAIN ANY COMMERCIAL PURPOSE IN ANY WAY.
