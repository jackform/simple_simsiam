import os.path

import torch.nn as nn
import torch


class Residual_Block(nn.Module):
    def __init__(self, ic, oc, stride=1):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oc),
        )

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or (ic != oc):
            self.downsample = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(oc),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class Classifier(nn.Module):
    def __init__(self, block=Residual_Block, num_layers=[2, 4, 3, 1], num_classes=11):
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer0 = self.make_residual(block, 32, 64, num_layers[0], stride=2)
        self.layer1 = self.make_residual(block, 64, 128, num_layers[1], stride=2)
        self.layer2 = self.make_residual(block, 128, 256, num_layers[2], stride=2)
        self.layer3 = self.make_residual(block, 256, 512, num_layers[3], stride=2)

        # self.avgpool = nn.AvgPool2d(2)
        self.prefc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512, 11)

    def make_residual(self, block, ic, oc, num_layer, stride=1):
        layers = []
        layers.append(block(ic, oc, stride))
        for i in range(1, num_layer):
            layers.append(block(oc, oc))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input size:", x.size())
        # [3, 128, 128]
        out = self.preconv(x)  # [32, 64, 64]
        out = self.layer0(out)  # [64, 32, 32]
        out = self.layer1(out)  # [128, 16, 16]
        out = self.layer2(out)  # [256, 8, 8]
        out = self.layer3(out)  # [512, 4, 4]
        # out = self.avgpool(out) # [512, 2, 2]
        preout = self.prefc(out.view(out.size(0), -1))
        out = self.fc(preout)
        return out


def resNetFood():
    model = Classifier()
    model_path = "./sample_3_best.ckpt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print('no checkpoint exists')

    return model
