import torch
import torch.nn as nn

def doubleConv2d(in_c, out_c, kernel_size = 3):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU()
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.maxPool_2x2 = nn.MaxPool2d(kernel_size=2)
        self.doubleConv_1 = doubleConv2d(1, 64)
        self.doubleConv_2 = doubleConv2d(64, 128)
        self.doubleConv_3 = doubleConv2d(128, 256)
        self.doubleConv_4 = doubleConv2d(256, 512)
        self.doubleConv_5 = doubleConv2d(512, 1024)

    def forward(self, x):
        x1 = self.doubleConv_1(x)
        x1 = self.maxPool_2x2(x1)
        x2 = self.doubleConv_2(x1)
        x2 = self.maxPool_2x2(x2)
        x3 = self.doubleConv_3(x2)
        x3 = self.maxPool_2x2(x3)
        x4 = self.doubleConv_4(x3)
        x4 = self.maxPool_2x2(x4)
        x5 = self.doubleConv_5(x4)
        print(x5.size())


if __name__ == "__main__":
    x = torch.rand((1, 1, 572, 572))
    model = UNet()
    model(x)
