import torch.nn as nn
import torch

class SmallerWitness(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)   
        self.pool1 = nn.MaxPool2d(2)      

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.flatten(x)      
        output = torch.softmax(self.fc1(x), dim=1)

        return output    