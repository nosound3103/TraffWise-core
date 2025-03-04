import cv2
import numpy as np
import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort


class ViewTransformer:
    def __init__(self, source, target):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.matrix)

        return transformed_points.reshape(-1, 2)


class DeepSORT:
    def __init__(self,
                 config,
                 camera_name,
                 max_age=15):

        self.config = config
        self.annotation_path = self.config["samples"][camera_name]["annotation_path"]
        self.read_annotation()

        self.view_transformer = ViewTransformer(self.SOURCE, self.TARGET)

        self.tracker = DeepSort(
            max_age=max_age, max_iou_distance=0.90, n_init=2)

        self.class_names = list(config["labels"].keys())

    def read_annotation(self):
        """ Read annotation file """

        with open(self.annotation_path, "r") as file:
            self.data = yaml.safe_load(file)

        self.SOURCE = np.array(self.data["SOURCE"]["box"])
        self.TARGET = np.array(self.data["TARGET"]["box"])

    def convert_box(self, box):
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]

    def extract_detections(self, boxes, conf_threshold=0.5):
        detections = [
            [self.convert_box(box),
             conf,
             label]
            for [label, conf, box] in boxes if conf > conf_threshold]

        return detections

    def update_tracks(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
