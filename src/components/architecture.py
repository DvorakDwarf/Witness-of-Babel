import torch.nn as nn
import torch

class Witness(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)   
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)      
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)   
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)   

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3)   
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)   

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2304, 256)    
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.relu(self.conv3(x))
        # x = torch.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.relu(self.conv5(x))
        # x = torch.relu(self.conv6(x))
        x = self.pool3(x)

        x = self.flatten(x)      
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        output = torch.softmax(self.fc2(x), dim=1)

        return output    

# def __init__(self):
#         super().__init__()        
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(1, 16, kernel_size=5),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(16, 32, kernel_size=3),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )
#         self.conv3 = nn.Sequential(         
#             nn.Conv2d(32, 64, kernel_size=3),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )  