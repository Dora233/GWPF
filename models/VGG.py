import torch.nn as nn
import torch.nn.functional as F
import math

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)

class VGG11_EMNIST(nn.Module):
    def __init__(self, num_classes=47):
        super(VGG11_EMNIST, self).__init__()
        cfg=[32, 'M', 64, 'M', 128, 128, 'M', 128, 256, 'M', 256, 256, 'M']
        self.features = self.make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self,cfg, batch_norm=True):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
