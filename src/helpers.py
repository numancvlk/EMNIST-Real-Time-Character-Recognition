#SCRIPTS
from model import device

#LIBRARIES
import torch


def accuracy(yTrue,yPred):
    correct = torch.eq(yTrue,yPred).sum().item()
    acc = correct / len(yTrue)
    return acc

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Total train time is {totalTime} on the {device}")


def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              accFn,
              device:torch.device = device):
    
    trainLoss, trainAccuracy = 0,0
    model.train()

    for batch, (xTrain,yTrain) in enumerate(dataLoader):
        xTrain,yTrain = xTrain.to(device), yTrain.to(device)

        #1 - FORWARD
        trainPred = model(xTrain)

        #2 - LOSS / ACC
        loss = lossFn(trainPred,yTrain)
        trainLoss += loss

        acc = accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))
        trainAccuracy += acc

        #3 - ZERO GRAD
        optimizer.zero_grad()

        #4 - BACKWARD
        loss.backward()

        #5 - STEP
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(xTrain)} / {len(dataLoader.dataset)} samples")
    
    trainLoss /= len(dataLoader)
    trainAccuracy /= len(dataLoader)

    print(f"Train Loss: {trainLoss:.5f} | Train Accuracy: {trainAccuracy:.5f}%")


def testStep(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn,
             device: torch.device = device):
    
    testLoss, testAccuracy = 0,0
    model.eval()

    with torch.inference_mode():
        for xTest, yTest in dataLoader:
            
            xTest,yTest = xTest.to(device), yTest.to(device)

            #1 - FORWARD
            testPred = model(xTest)

            #2 - LOSS / ACC
            loss = lossFn(testPred,yTest)
            testLoss += loss.item()

            acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
            testAccuracy += acc

    testLoss /= len(dataLoader)
    testAccuracy /= len(dataLoader)

    print(f"Test Loss: {testLoss:.5f} | Test Accuracy: {testAccuracy:.5f}%")

def modelSummary(model: torch.nn.Module,
                 dataLoader: torch.utils.data.DataLoader,
                 lossFn: torch.nn.Module,
                 accFn,
                 device: torch.device = device):
    
    summaryLoss, summaryAccuracy = 0,0
    model.eval()

    with torch.inference_mode():
        for xTest, yTest in dataLoader:
            xTest, yTest = xTest.to(device), yTest.to(device)

            #1- FORWARD
            summaryPred = model(xTest)

            #2 - LOSS / ACC
            loss = lossFn(summaryPred,yTest)
            summaryLoss += loss.item()

            acc = accFn(yTrue = yTest, yPred = summaryPred.argmax(dim=1))
            summaryAccuracy += acc

    summaryLoss /= len(dataLoader)
    summaryAccuracy /= len(dataLoader)

    return {"MODEL NAME": model.__class__.__name__,
            "MODEL LOSS": summaryLoss,
            "MODEL ACCURACY": summaryAccuracy}

    