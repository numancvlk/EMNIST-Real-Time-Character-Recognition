#SCRIPTS
from model import myModel
from helpers import accuracy,trainStep,printTrainTime,testStep,modelSummary

#LIBRARIES
import torch
from torch import nn

LEARNING_RATE = 0.1

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=myModel.parameters(),
                            lr=LEARNING_RATE)




