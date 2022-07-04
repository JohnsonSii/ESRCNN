import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(),

        )
        self.head = nn.Sequential(
            nn.Conv2d(32, 3, 5, 1, 2),
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        head = self.head(conv2_out)
        # out = F.interpolate(head, (x.size()[2] * 2, x.size()[3] * 2), mode='bilinear', align_corners=True)
        return head


if __name__ == '__main__':
    net = Net()
    arr = torch.randn((1, 3, 64, 64))
    print(net(arr).size())
