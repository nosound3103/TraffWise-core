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


class Tracker:
    def __init__(self,
                 config,
                 camera_name,
                 max_age=15):

        self.config = config
        self.annotation_path = self.config["samples"][camera_name]["annotation_path"]
        self.read_annotation()

        self.view_transformer = ViewTransformer(self.SOURCE, self.TARGET)

        self.deep_sort = DeepSort(max_age=max_age)

        self.class_names = list(config["labels"].keys())

        self.colors = np.random.randint(
            0, 255, size=(len(self.class_names), 3))

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

    def draw_tracks(self, frame, tracks, speed_estimator):
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = int(track.get_det_class())

            x1, y1, x2, y2 = map(int, ltrb)
            color = self.colors[class_id]
            B, G, R = map(int, color)
            label = f"{self.class_names[class_id]}-{track_id}"

            x_center = (ltrb[0] + ltrb[2]) / 2
            y_center = ltrb[3]

            transformed_point = self.view_transformer.transform_points(
                np.array([[x_center, y_center]]))[0]

            speed_estimator.update_coordinates(
                track_id, transformed_point)
            speed = speed_estimator.calculate_speed(track_id)

            label += f" {int(speed)} km/h"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
