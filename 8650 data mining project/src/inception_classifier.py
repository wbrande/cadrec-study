import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__()

        self.branch1 = nn.Conv3d(in_channels, out_1x1, kernel_size=(1, 1, 1))

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, red_3x3, kernel_size=(1, 1, 1), padding=1),
            nn.Conv3d(red_3x3, red_3x3, kernel_size=(3, 1, 1)),
            nn.Conv3d(red_3x3, red_3x3, kernel_size=(1, 3, 1)),
            nn.Conv3d(red_3x3, out_3x3, kernel_size=(1, 1, 3)),
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, red_5x5, kernel_size=(1, 1, 1), padding=2),
            nn.Conv3d(red_5x5, red_5x5, kernel_size=(5, 1, 1)),
            nn.Conv3d(red_5x5, red_5x5, kernel_size=(1, 5, 1)),
            nn.Conv3d(red_5x5, out_5x5, kernel_size=(1, 1, 5)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_1x1pool, kernel_size=(1, 1, 1)),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_func = nn.BCELoss()
        self.conv1 = nn.Conv3d(1, 32, 9, 2)
        self.conv2 = nn.Conv3d(32, 16, 7, 2)
        self.inception = Inception(16, 4, 8, 16, 4, 8, 4)
        self.conv3 = nn.Conv3d(32, 8, 3, 2)
        # fully connected layers
        self.fc1 = nn.Linear(8*14*14*8, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv3d expect input [batch_size, channels, depth, height, width].
        # x batch, width, height, depth
        x = x.permute(0, 3, 2, 1).unsqueeze(1)
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.inception(x))
        x = F.tanh(self.conv3(x))

        x = F.max_pool3d(x, 2)
        x = x.view(x.shape[0], -1)

        x = torch.sigmoid(self.fc1(x))

        return x.squeeze(1)

    def loss(self, pred, gt):
        return self.loss_func(pred, gt)
