#SCRIPTS
from model import EMNISTModel, device
from helpers import accuracy,trainStep,printTrainTime,testStep,modelSummary
from dataset import trainDataLoader,testDataLoader,trainData

#LIBRARIES
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from tqdm.auto import tqdm

from timeit import default_timer

import random

LEARNING_RATE = 0.003
print(len(trainData.classes))
myModel = EMNISTModel(inputShape=1, hiddenUnit=128, outputShape=len(trainData.classes)).to(device)

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=myModel.parameters(),
                            lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)


random.seed(30)
epochs = 12

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
    
    scheduler.step()

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
