import torch
from torch import nn
import torch.nn.functional as F
import pdb
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(527, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input1, input2):
        x = torch.abs(input1-input2)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

class SiameseNetworkConcat(nn.Module):
    def __init__(self):
        super(SiameseNetworkConcat, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(1054, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
#         self.siamese = nn.Sequential(
#             nn.Linear(1054, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#         )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input1, input2):
        input1 = torch.reshape(input1, (input1.shape[0], -1))
        input2 = torch.reshape(input2, (input2.shape[0], -1))
#         input1 = self.siamese(input1)
#         input2 = self.siamese(input2)
        x = torch.cat((input1, input2), 1)
        #x = torch.abs(input1-input2)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x
