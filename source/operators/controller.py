import cv2
import numpy as np
from typing import Literal, Dict

from ..engines import VehicleDetector, Tracker, \
    RedLightViolationDetector, SpeedEstimator


class Controller:
    def __init__(self,
                 config,
                 camera_name:
                     Literal[
                         "speed_test",
                         "speed_test_1",
                         "red_light_violation_test",
                         "camera_1_night",
                         "camera_2_night",
                         "camera_3_night",
                         "camera_1_day",
                         "camera_2_day"] = "red_light_violation_test"):
        self.config = config
        self.camera_name = camera_name
        self.vehicle_detector = VehicleDetector(config["models"])
        self.tracker = Tracker(
            config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(
            config, self.camera_name)
        self.speed_estimator = SpeedEstimator()

    def process_frame(self, frame, features: Dict[str, bool] = {
            "speed_estimation": True,
            "red_light_violation_detection": True,
            "wrong_lane_driving_detection": True,
    }):
        """ Process frame

        Args:
            frame (np.array): Frame of video
        """

        boxes = self.vehicle_detector.detect(frame)

        detections = self.tracker.extract_detections(boxes)
        tracks = self.tracker.deep_sort.update_tracks(
            detections, frame=frame)

        if features["speed_estimation"]:
            self.tracker.draw_tracks(
                frame, tracks, self.speed_estimator)

        if features["red_light_violation_detection"]:
            frame, detected_color = self.rlv_detector.detect_traffic_light_color(
                frame)
            frame, violation = self.rlv_detector.detect_red_light_violation(
                frame, tracks)

        if features["wrong_lane_driving_detection"]:
            pass

        return frame

    def process_video(self, features: Dict[str, bool] = {
            "speed_estimation": True,
            "red_light_violation_detection": True,
            "wrong_lane_driving_detection": True,
    }):
        """ Process video """
        video_path = self.config["samples"][self.camera_name]["video_path"]
        output_path = self.config["samples"][self.camera_name]["output_path"]
        frame_size = (1280, 960)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.speed_estimator.fps = fps
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while True:

            ret, frame = cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            frame = self.process_frame(frame, features)
            resized_frame = cv2.resize(frame, frame_size)

            out.write(resized_frame)

            cv2.imshow("Video Frame", resized_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
