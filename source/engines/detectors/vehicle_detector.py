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
    def __init__(self,
                 config,
                 model_type: Literal["yolo11", "rtdetrv2", "faster_rcnn"] = "yolo11"):
        self.config = config
        self.model_type = model_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.iou_threshold = None
        self.conf_threshold = None

        self.load_model()

    def load_model(self):
        self.iou_threshold = self.config["models"][self.model_type].get(
            "iou_threshold", 0.5)
        self.conf_threshold = self.config["models"][self.model_type].get(
            "conf_threshold", 0.5)

        if self.model_type == "yolo11":
            self.model = YOLO(self.config["models"][self.model_type]["path"])
        elif self.model_type == "rtdetrv2":
            self.model = RTDETR(self.config["models"][self.model_type]["path"])
        elif self.model_type == "faster_rcnn":
            self.model = FRCNN(num_classes=5)

            checkpoint = torch.load(
                self.config["models"][self.model_type]["path"],
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

        if self.model_type in ["yolo11", "rtdetrv2"]:
            boxes = predictions[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = predictions[0].boxes.conf.cpu().numpy()
            labels = predictions[0].boxes.cls.cpu().numpy().astype(int)

        elif self.model_type == "faster_rcnn":
            confs = predictions["scores"]
            labels = predictions["labels"]
            boxes = predictions["boxes"]

            conf_mask = confs > self.conf_threshold
            boxes = boxes[conf_mask]
            labels = labels[conf_mask]
            confs = confs[conf_mask]

            keep = nms(boxes, confs, iou_threshold=self.iou_threshold)
            boxes = boxes[keep]
            labels = labels[keep]
            confs = confs[keep]

            boxes = scale_bboxes(boxes.cpu().numpy(),
                                 (self.width, self.height))
            labels = labels.cpu().numpy() - 1  # Adjust for background class
            confs = confs.cpu().numpy()

        res = list(zip(labels, confs, boxes))

        return res

    def detect(self, image):
        if self.model_type in ["yolo11", "rtdetrv2"]:
            predictions = self.model.predict(
                image,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold)

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
