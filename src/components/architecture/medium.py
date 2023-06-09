#Good enough size for me personally
#Still simple and quick cuz fc1 small

import torch.nn as nn
import torch

class MediumWitness(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)   
        self.pool1 = nn.MaxPool2d(2)      
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)   
        self.pool2 = nn.MaxPool2d(2)   

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)      
        output = torch.softmax(self.fc1(x), dim=1)

        return output    