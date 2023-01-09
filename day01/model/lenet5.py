import torch
import torch.nn as nn

class Lenet5(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train=False) -> None:
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_width = in_width
        self.in_height = in_height
        self.in_channel = in_channel
        self.is_train = is_train

        # Convolution Output : [(W - K + 2P) / S] + 1
        # W : 1-channel size of input data (e.g. 32 X 32 -> 32)
        # K : Kernel size
        # P : Pooling
        # S : Size
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        # Fully-Conntected Layer
        self.fc3 = nn.Linear(120, 84)           # size 84
        self.fc4 = nn.Linear(84, self.n_classes)    # size 10

    def forward(self, x):
        # Shape of x : [B, C, H, W]
        x = self.conv0(x)
        x = torch.tanh(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = torch.tanh(x)

        # Change format from 4-dimension to 2-dimension ([B, C, H, W] -> [B, C * H * W])
        x = torch.flatten(x, start_dim=1)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = x.view(self.batch, -1)
        x = nn.functional.softmax(x, dim=1)
        
        if self.is_train is False:
            x = torch.argmax(x, dim=1)
        return x
         

        