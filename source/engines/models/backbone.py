import torch.nn as nn
import torchvision.models as models


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2, self).__init__()
        mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet_v2.features

    def forward(self, x):
        x = self.features(x)
        return x


class MobileNetV3(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3, self).__init__()
        mobilenet_v3 = models.mobilenet_v3_large(pretrained=pretrained)
        self.features = mobilenet_v3.features

    def forward(self, x):
        x = self.features(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SqueezeNet, self).__init__()
        squeezenet = models.squeezenet1_1(pretrained=pretrained)
        self.features = squeezenet.features

    def forward(self, x):
        x = self.features(x)
        return x


class TinyDetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(TinyDetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2)
        self.layer3 = self._make_layer(64, 128, 2)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
