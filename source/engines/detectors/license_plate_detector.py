import torch
import numpy as np
from ultralytics import YOLO
from typing import Dict


class LicensePlateDetector:
    """
    License plate detector using YOLO model to detect and extract license plates from vehicle images.
    """

    def __init__(self, config: Dict):
        """
        Initialize the license plate detector.

        Args:
            config: Configuration dictionary containing model paths and parameters
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = 0.5

        self.model = YOLO(self.config["lp"]["path"])

    def detect_license_plates(self, frame, box):
        """
        Detect license plates in vehicle crops.

        Args:
            frame: Full frame from video
            vehicle_boxes: List of vehicle bounding boxes [x1, y1, x2, y2]

        Returns:
            List of dictionaries containing license plate detections with vehicle IDs
        """
        x1, y1, x2, y2 = map(int, box)
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        vehicle_crop = frame[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            return [0, 0, 0, 0]

        detections = self.model(
            vehicle_crop, conf=self.confidence_threshold, verbose=False)
        boxes = detections[0].boxes

        if len(boxes) == 0:
            return [0, 0, 0, 0]

        confs = boxes.conf.cpu().numpy()
        max_conf_idx = np.argmax(confs)

        if max_conf_idx < self.confidence_threshold:
            return [0, 0, 0, 0]

        box = boxes[max_conf_idx]
        lp_x1, lp_y1, lp_x2, lp_y2 = box.xyxy[0].cpu().numpy().astype(int)
        lp_x1 += x1
        lp_y1 += y1
        lp_x2 += x1
        lp_y2 += y1

        return [lp_x1, lp_y1, lp_x2, lp_y2]
