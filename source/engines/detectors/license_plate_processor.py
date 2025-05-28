import torch
import numpy as np
import re
import cv2
import easyocr
from ultralytics import YOLO
from paddleocr import PaddleOCR


class LicensePlateProcessor:
    def __init__(self, config, ocr_type="easyocr"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.config = config or {}
        self.ocr_type = ocr_type

        if config:
            self.detection_model = YOLO(
                self.config["models"]["lp_detector"]["path"])
            self.conf_threshold = self.config["models"]["lp_detector"].get(
                "conf_threshold", 0.5)

        if ocr_type == "easyocr":
            self.recognition_model = easyocr.Reader(['en'], gpu=True)
        elif ocr_type == "paddleocr":
            self.recognition_model = PaddleOCR(
                lang="en",
                ocr_version="PP-OCRv4")

    def add_padding(self, frame):
        padded_frame = cv2.copyMakeBorder(
            frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT,
            value=[255, 255, 255])

        return padded_frame

    def detect_license_plate(self, frame, box):
        x1, y1, x2, y2 = box
        vehicle_crop = frame[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            return [0, 0, 0, 0]

        detections = self.detection_model(
            vehicle_crop, conf=self.conf_threshold, verbose=False)
        detected_boxes = detections[0].boxes

        if len(detected_boxes) == 0:
            return [0, 0, 0, 0]

        confs = detected_boxes.conf.cpu().numpy()
        max_conf_idx = np.argmax(confs)

        detected_box = detected_boxes[max_conf_idx]
        lp_x1, lp_y1, lp_x2, lp_y2 = detected_box.xyxy[0].cpu(
        ).numpy().astype(int)
        lp_x1 += x1
        lp_y1 += y1
        lp_x2 += x1
        lp_y2 += y1

        return [lp_x1, lp_y1, lp_x2, lp_y2]

    def detect_license_plates(self, frame, boxes):
        results = []
        for box in boxes:
            lp_box = self.detect_license_plate(frame, box)
            results.append(lp_box)
        return results

    def extract_text(self, frame, box=None):
        if box:
            x1, y1, x2, y2 = box
            lp_crop = frame[y1:y2, x1:x2]

            if lp_crop.size == 0:
                return 'unknown', 0.0
        else:
            lp_crop = frame.copy()

        lp_crop = self.add_padding(lp_crop)

        if self.ocr_type == "easyocr":
            results = self.recognition_model.readtext(lp_crop)
            text = "".join([res[1] for res in results if res[1]])
            conf = sum([res[2] for res in results]) / \
                len(results) if results else 0
        elif self.ocr_type == "paddleocr":
            result = self.recognition_model.predict(lp_crop)
            if not len(result[0]['rec_texts']):
                return 'unknown', 0.0

            text = "".join(result[0]['rec_texts'])
            conf = sum(result[0]['rec_scores']) / \
                len(result[0]['rec_scores']) if result[0]['rec_scores'] else 0

        if text:
            return text, conf
        else:
            return "unknown", 0.0

    def extract_texts(self, frame, boxes):
        plate_texts = []
        for box in boxes:
            plate_text = self.extract_text(frame, box)
            plate_texts.append(plate_text)
        return plate_texts

    def validate_license_plate(self, plate_text):
        if plate_text == "unknown":
            return False

        pattern = r'^\d{2}[A-HK-NP-Z]-\d{3}\.\d{2}$'
        return re.match(pattern, plate_text.upper()) is not None
