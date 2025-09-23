#LIBRARIES
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

trainTransform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0,(0.1,0.1)),
    transforms.RandomResizedCrop(28, scale=(0.9,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#COLLECTING DATAS
trainData = datasets.EMNIST(
    root="datas",
    split="byclass",
    train=True,
    download=True,
    transform = trainTransform
)

testData = datasets.EMNIST(
    root="datas",
    split="byclass",
    train=False,
    download=True,
    transform = testTransform
)

#VERİLERİ GÖRSELLEŞTİRMEK
fig = plt.figure(figsize=(10,10))
rows,cols = 5,5

for i in range(1,rows*cols+1):
    randomIndex = torch.randint(0,len(trainData),size=[1]).item()
    image, label = trainData[randomIndex]
    ax = plt.subplot(rows,cols,i)
    ax.set_title(trainData.classes[label])
    ax.imshow(image.squeeze())
    ax.axis("off")
plt.show()

print(image.shape)

BATCH_SIZE = 32

trainDataLoader = DataLoader(
    dataset=trainData,
    batch_size= BATCH_SIZE,
    shuffle=True
)

testDataLoader = DataLoader(
    dataset=testData,
    batch_size=BATCH_SIZE,
    shuffle=False
)