import torch.nn as nn
import torch
class CNN(nn.Module):
    def __init__(self, numclasses):
        super().__init__()
        self.conv1 = self.block(3, 16)
        self.conv2 = self.block(16, 32)
        self.conv3 = self.block(32, 64)
        self.conv4 = self.block(64,128)
        self.conv5 = self.block(128, 128)

        self.fcl1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features= 6272, out_features=512),
            nn.ReLU()
        )
        self.fcl2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features= 512, out_features=256),
            nn.ReLU()
        )

        self.fcl3 = nn.Linear(in_features= 256, out_features=numclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        b, c, h, w = x.shape    # batch, channel, height, width
        x= x.view(b, -1)        # flatten the tensor     

        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.fcl3(x)

        return x # the format of x is [batch, num_classes]
        
    def block ( self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 2x2 max pooling and stride = kernel_size
        )
