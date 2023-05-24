import torch.nn as nn

class Witness(nn.module):
    def __init__(self):
        super(Witness(), self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(1, 8, kernel_size=3),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, kernel_size=3),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 5 * 5, 10)    
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        
        x = self.flatten(x)       
        output = self.fc1(x)
        return output    