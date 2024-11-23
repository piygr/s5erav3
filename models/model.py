import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*4*4, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  #in: 28x28, out: 28x28
        x = self.dropout(x)
        x = self.pool(x)                        #in: 28x28, out: 14x14
        x = self.relu(self.bn2(self.conv2(x)))  #in: 14x14, out: 12x12
        x = self.dropout(x)
        x = self.pool(x)                        #in: 12x12, out: 6x6
        x = self.conv3(x)                       #in: 6x6, out: 4x4
        x = x.view(-1, 16*4*4)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
