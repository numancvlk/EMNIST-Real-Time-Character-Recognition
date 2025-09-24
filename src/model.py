#SCRIPTS
from dataset import trainData

#LIBRARIES
import torch
from torch import nn

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
    
device = "cuda" if torch.cuda.is_available() else "cpu"