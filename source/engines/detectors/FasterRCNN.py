import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import os
from torchvision.ops import nms
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import mobilenet_v3_small
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import MobileNet_V3_Small_Weights
import cv2


class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self):
        super().__init__(min_size=640, max_size=640, image_mean=[
            0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def resize(self, image, target):
        image = F.resize(image, [640, 640])

        if target is not None and "boxes" in target:
            w_old, h_old = image.shape[-1], image.shape[-2]
            w_new, h_new = 640, 640
            scale_w = w_new / w_old
            scale_h = h_new / h_old
            target["boxes"][:, [0, 2]] *= scale_w
            target["boxes"][:, [1, 3]] *= scale_h
        return image, target


class FRCNN(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained=MobileNet_V3_Small_Weights.DEFAULT):
        super(FRCNN, self).__init__()
        self.num_classes = num_classes
        self.backbone = self.get_backbone(pretrained)

        self.anchor_sizes = (32, 64, 128, 256)
        self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.anchor_sizes)

        self.anchor_generator = AnchorGenerator(
            sizes=self.anchor_sizes,
            aspect_ratios=self.aspect_ratios
        )

        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=self.anchor_generator
        )

        self.model.transform = CustomRCNNTransform()

    def get_backbone(self, pretrained):
        backbone = mobilenet_v3_small(weights=pretrained).features
        return_layers = {'2': '0', '7': '1', '12': '2'}
        in_channels = [24, 48, 576]

        backbone.out_channels = 64
        fpn = BackboneWithFPN(
            backbone=backbone,
            return_layers=return_layers,
            in_channels_list=in_channels,
            out_channels=64
        )

        return fpn

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            return self.model(images, targets)
        else:
            return self.model(images)


def detection_collate_fn(batch):
    return tuple(zip(*batch))
