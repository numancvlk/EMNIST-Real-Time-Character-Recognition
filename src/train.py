#SCRIPTS
from model import EMNISTModel, device
from helpers import accuracy,trainStep,printTrainTime,testStep,modelSummary
from dataset import trainDataLoader,testDataLoader,trainData

#LIBRARIES
import torch
from torch import nn

from tqdm.auto import tqdm

from timeit import default_timer

import random

LEARNING_RATE = 0.01
print(len(trainData.classes))
myModel = EMNISTModel(inputShape=1, hiddenUnit=32, outputShape=len(trainData.classes)).to(device)

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=myModel.parameters(),
                            lr=LEARNING_RATE)


random.seed(30)
epochs = 5

startTrainTimer = default_timer()

for epoch in tqdm(range(epochs)):
    trainStep(model=myModel,
              dataLoader=trainDataLoader,
              optimizer=optimizer,
              lossFn=lossFn,
              accFn=accuracy,
              device=device)
    
    testStep(model=myModel,
             dataLoader=testDataLoader,
             lossFn=lossFn,
             accFn=accuracy,
             device=device)

endTrainTimer = default_timer()

printTrainTime(start=startTrainTimer,
               end=endTrainTimer,
               device=device)

modelResult = modelSummary(model=myModel,
                           dataLoader=testDataLoader,
                           lossFn=lossFn,
                           accFn=accuracy,
                           device=device)
print(modelResult)


torch.save(myModel.state_dict(),"myEMNISTMODEL.pth")
print("Model ağırlıkları kaydedildi!")
