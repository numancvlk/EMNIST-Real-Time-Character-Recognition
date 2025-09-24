#SCRIPTS
from model import device
from dataset import trainData

#LIBRARIES
import cv2 as cv

import torch
from torch import nn

from torchvision import transforms

from collections import deque



class EMNISTModel(nn.Module):
    def __init__(self,
                 inputShape:int,
                 hiddenUnit:int,
                 outputShape:int):
        super().__init__()

        self.convStack1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.convStack2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnit,
                      out_channels=hiddenUnit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnit* 7 * 7,
                      out_features=outputShape)
        )

    def forward(self,x):
        x = self.convStack1(x)
        x = self.convStack2(x)
        x = self.classifier(x)
        return x

trainedModel = EMNISTModel(inputShape=1,
                           hiddenUnit=128,
                           outputShape=len(trainData.classes)).to(device)

trainedModel.load_state_dict(torch.load("src\myEMNISTMODEL.pth", map_location=device))

trainedModel.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

predictionQueue = deque(maxlen=10)
confidenceThreshold = 0.7
frameCount = 0

videoCapture = cv.VideoCapture(0)
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 1)

while True:
    isTrue, frame = videoCapture.read()

    if not isTrue:
        break
    
    grayFrame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    thresh = cv.morphologyEx(
    cv.threshold(cv.GaussianBlur(grayFrame, (5,5), 0), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1],
    cv.MORPH_OPEN,
    cv.getStructuringElement(cv.MORPH_RECT, (3,3))
)

    contours = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]

    if contours:
        cnt = max(contours, key=cv.contourArea)

        if 2000 < cv.contourArea(cnt) < 60000:
        # x, y, w, h = cv2.boundingRect(cnt) OpenCV’de bir konturu (çizgilerle çevrili şekli) dikdörtgen içine saran en küçük dikdörtgeni hesaplamak için kullanılır.
        # x → Dikdörtgenin sol üst köşesinin x koordinatı
        # y → Dikdörtgenin sol üst köşesinin y koordinatı
        # w → Dikdörtgenin genişliği (width)
        # h → Dikdörtgenin yüksekliği (height)
            x,y,w,h = cv.boundingRect(cnt)
            
            #REGION OF INTEREST
            roi = cv.resize(grayFrame[y:y+h, x:x+w], (28,28)) # Resmin bir kısmını alıp (ROI → Region of Interest / İlgi Alanı) modelin beklediği boyuta getirmek.
            roiTensor = transform(roi).unsqueeze(0).to(device) #ROI’yi PyTorch modelinin anlayabileceği tensöre dönüştürmek

            if frameCount % 2== 0:
                with torch.no_grad(): #Tahmin (inference) sırasında gradyan hesaplamalarını kapatmak.
                    probs = torch.softmax(trainedModel(roiTensor), dim=1)
                    #predClass → Tahmin edilen sınıfın indeksi
                    #predConfidence → Tahminin olasılığı (güven skoru)
                    predConfidence, predClass = torch.max(probs,dim=1) #en yüksek olasılığa sahip sınıfı bulur
                    predConfidence, predClass = predConfidence.item(), predClass.item() #.item() → tek elemanlı tensor’ları Python float veya int değerine çevirir

                if predConfidence >= confidenceThreshold:
                    predictionQueue.append(predClass) #EĞER GÜVENİLİRLİK SKORU BİZİM BAŞTA BELİRLEDİĞİMİZ GÜVENİLİRLİK SKORUNDAN YÜKSEKSE TAHMİNİ TAHMİN KUYRUĞUNA EKLE
                    stablePrediction = max(set(predictionQueue), key=predictionQueue.count) #SON 5 TAHMİNDEN EN ÇOK HANGİSİ TAHMİN EDİYORSA ONU AL
                    stablePredictionChar = trainData.classes[stablePrediction]
                    # (x, y) → dikdörtgenin sol üst köşesi
                    # (x+w, y+h) → dikdörtgenin sağ alt köşesi

                    cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
                    cv.putText(frame,f"Pred = {stablePredictionChar} ({predConfidence*100:.0f}%)", (x,y-10), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
        cv.imshow("Camera", frame)
        frameCount += 1

        if cv.waitKey(10) == 27:
            break
            

videoCapture.release()
cv.destroyAllWindows()