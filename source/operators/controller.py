import cv2
import numpy as np
from ..engines import VehicleDetector, RedLightViolationDetector


class Controller:
    def __init__(self, config):
        self.config = config
        # self.rlv_detector = RedLightViolationDetector(config)
        self.vehicle_detector = VehicleDetector(config["models"])

    def process_frame(self, frame):
        """ Process frame

        Args:
            frame (np.array): Frame of video
        """
        frame, detected_color = self.rlv_detector.detect_traffic_light_color(
            frame)
        frame, boxes = self.vehicle_detector.detect(frame)

        if detected_color == "red":
            frame, violation = self.rlv_detector.detect_red_light_violation(
                frame, boxes)
        else:
            violation = False

        return frame, violation

    def test(self, image):
        return self.vehicle_detector.detect(image)
