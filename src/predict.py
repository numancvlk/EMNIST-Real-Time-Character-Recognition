#SCRIPTS
from model import device, EMNISTModel
from dataset import testData

#LIBRARIES
import torch
import random
import matplotlib.pyplot as plt

newModel = EMNISTModel()

newModel.load_state_dict(torch.load("myEMNISTMODEL.pth"))

def makePredictions(model:torch.nn.Module,
                    data:list,
                    device:torch.device=device):

  predProbs = []
  model.to(device)
  model.eval()

  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample,dim=0).to(device)

      predLogits = model(sample)

      predProb = torch.softmax(predLogits.squeeze(), dim=0)

      predProbs.append(predProb.cpu())

  return torch.stack(predProbs)



random.seed(41)
testSamples = []
testLabels = []

for sample, label in random.sample(list(testData), k=9):
  testSamples.append(sample)
  testLabels.append(label)


prediction = makePredictions(model=newModel,
                             data=testSamples,
                             device=device)

predictionClasses = prediction.argmax(dim=1)


# PLOT PREDICTIONS WITH COLORS
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(testSamples):
    plt.subplot(nrows, ncols, i+1)

    # Görüntüyü göster
    plt.imshow(sample.squeeze(), cmap="gray")
    plt.axis('off')  # Eksenleri gizle

    # Tahmin ve gerçek sınıf isimleri
    predLabel = testData.classes[predictionClasses[i]]  # modelin tahmini
    trueLabel = testData.classes[testLabels[i]]         # gerçek sınıf

    # Renk belirle
    color = "green" if predLabel == trueLabel else "red"

    # Başlık ekle
    plt.title(f"P: {predLabel}\nT: {trueLabel}", color=color, fontsize=10)

plt.tight_layout()
plt.show()

