import os
import torch
import cv2
import albumentations as A
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO, RTDETR
from torchvision import transforms
from api.source.engines.detectors.FasterRCNN import FRCNN
from api.utils import scale_bboxes
from typing import Literal


class VehicleDetector:
    def __init__(self, config, model_type: Literal["yolo11", "rtdetrv2", "faster_rcnn"] = "yolo11"):
        self.config = config
        self.model_type = model_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()

    def load_model(self):
        if self.model_type == "yolo11":
            self.model = YOLO(self.config[self.model_type]["path"])
        elif self.model_type == "rtdetrv2":
            self.model = RTDETR(self.config[self.model_type]["path"])
        elif self.model_type == "faster_rcnn":
            self.model = FRCNN(num_classes=5)

            checkpoint = torch.load(
                self.config[self.model_type]["path"],
                map_location=self.device,
                weights_only=True)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(640, 640)),
            ])

    def preprocess_image(self, image):
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        self.width, self.height = image_pil.size

        image_tensor = self.data_transform(
            image_pil).unsqueeze(0).to(self.device)
        return image_tensor

    def postprocess(self, predictions):
        # Postprocess
        res = []
        if self.model_type in ["yolo11", "rtdetrv2"]:
            confs = predictions[0].boxes.conf.cpu().numpy()
            labels = predictions[0].boxes.cls.cpu().numpy()
            boxes = predictions[0].boxes.xyxy.cpu().numpy()
            boxes = [[int(value) for value in box] for box in boxes]

            res = [[label, conf, box]
                   for label, conf, box in zip(labels, confs, boxes)]

        elif self.model_type == "faster_rcnn":
            confs = predictions["scores"]
            labels = predictions["labels"]
            boxes = predictions["boxes"]

            keep = nms(boxes, confs, iou_threshold=0.5)
            confs = confs[keep].cpu().numpy()
            labels = labels[keep].cpu().numpy()
            boxes = boxes[keep].cpu().numpy()
            boxes = scale_bboxes(boxes, (self.width, self.height))
            res = [[label - 1, conf, box]
                   for label, conf, box in zip(labels, confs, boxes)
                   if conf > 0.5]

        return res

    def detect(self, image):
        if self.model_type in ["yolo11", "rtdetrv2"]:
            predictions = self.model.predict(image, conf=0.5, verbose=False)

        elif self.model_type == "faster_rcnn":
            image = self.preprocess_image(image)

            with torch.no_grad():
                predictions = self.model(image)[0]

        # Postprocess
        res = self.postprocess(predictions)

        return res

    def switch_model(self, model_type: Literal["yolo11", "rtdetrv2", "faster_rcnn"]):
        if model_type == self.model_type:
            return

        self.model_type = model_type
        del self.model

        self.load_model()
