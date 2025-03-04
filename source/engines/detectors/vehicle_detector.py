import os
from ultralytics import YOLO, RTDETR
from typing import Literal


class VehicleDetector:
    def __init__(self, config, model_type: Literal["yolo11", "detr", "faster_rcnn"] = "yolo11"):
        self.config = config
        self.model_type = model_type

        if model_type == "yolo11":
            self.model = YOLO(self.config[model_type]["path"])
        elif model_type == "detr":
            self.model = RTDETR(self.config[model_type]["path"])
        # elif model_type == "faster_rcnn":
        #     self.model = FasterRCNN(self.config[model_type]["path"])

    def preprocess_image(self, image):
        # Preprocess image
        pass

    def postprocess(self, predictions):
        # Postprocess
        if self.model_type in ["yolo11", "detr"]:
            confs = predictions[0].boxes.conf.cpu().numpy()
            labels = predictions[0].boxes.cls.cpu().numpy()
            boxes = predictions[0].boxes.xyxy.cpu().numpy()
            boxes = [[int(value) for value in box] for box in boxes]

            res = [[label, conf, box]
                   for label, conf, box in zip(labels, confs, boxes)]

        elif self.model_type == "FasterRCNN":
            pass

        return res

    def detect(self, image):
        # Preprocess image
        if self.model_type == "FastRCNN":
            image = self.preprocess_image(image, imgsz=640)

        # Predict
        predictions = self.model.predict(image, conf=0.5, verbose=False)

        # Postprocess
        res = self.postprocess(predictions)

        return res
