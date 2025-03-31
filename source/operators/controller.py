import cv2
import time
import numpy as np
from typing import Literal, Dict
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon

from .adaptive_frame_skipper import AdaptiveFrameSkipper
from ..engines import VehicleDetector, DeepSORT, LaneManager, \
    RedLightViolationDetector, SpeedEstimator, WrongLaneDrivingDetector
from ..process import AsyncCloudinaryUploader


class Controller:
    def __init__(self,
                 config,
                 model_type:
                     Literal[
                        "yolo11",
                        "detr",
                        "faster_rcnn"] = "yolo11",
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
        self.class_names = list(config["labels"].keys())
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.colors = [
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (192, 192, 192)
        ]
        self.camera_name = camera_name
        self.frame_skipper = AdaptiveFrameSkipper(config["frame_skipper"])
        self.uploader = AsyncCloudinaryUploader()
        self.vehicle_detector = VehicleDetector(config["models"], model_type)
        self.tracker = DeepSORT(config, self.camera_name)
        self.lane_manager = LaneManager(config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(self.lane_manager, self.uploader)
        self.wrong_way = WrongLaneDrivingDetector(self.lane_manager, self.uploader)
        self.speed_estimator = SpeedEstimator(self.lane_manager, self.uploader)
        

    def draw_track(self, frame, log):
        x1, y1, x2, y2 = map(int, log["ltrb"])
        color = self.colors[log["class_id"]]
        B, G, R = map(int, color)

        if log["speed_violation"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
            cv2.putText(frame, "Over speed", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return
        if log["red_light_violation"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
            cv2.putText(frame, "RLV", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return
        if log["wrong_way_violation"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
            cv2.putText(frame, "Wrong_way", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return
        
        label = f"{self.class_names[log["class_id"]]}-{log["track_id"]}-{int(log["speed"])}-km/h"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, log["turn_type"], (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    
    def submit_violation(self, frame, log):
        if log["speed_violation"]:
            print(f"Track_id: {log['track_id']} got over speed violation!")
            frame_copy = frame.copy()
            future = self.executor.submit(self.speed_estimator.capture_violation, frame_copy, log)

        if log["red_light_violation"]:
            print(f"Track_id: {log['track_id']} got red light violation!")
            frame_copy = frame.copy()
            future = self.executor.submit(self.rlv_detector.capture_violation, frame_copy, log)

        if log["wrong_way_violation"]:
            print(f"Track_id: {log['track_id']} got wrong way violation!")
            frame_copy = frame.copy()
            future = self.executor.submit(self.wrong_way.capture_violation, frame_copy, log)


    def process_frame(self, frame):
        """ Process frame

        Args:
            frame (np.array): Frame of video
        """
        boxes = self.vehicle_detector.detect(frame)
        detections = self.tracker.extract_detections(boxes)
        tracks = self.tracker.update_tracks(detections, frame=frame)

        frame = self.rlv_detector.detect_traffic_light_color(frame)
        violation_frame = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = int(track.get_det_class())
            # Speed Estimate
            speed_violation, speed = self.speed_estimator.detect_speed(track_id, ltrb, frame)
            # Wrong way driving detection
            wrong_way_violation, turn_type = self.wrong_way.detect_violation(track_id, ltrb, frame, speed)
            # Red light detection
            red_light_violation = self.rlv_detector.detect_red_light_violation(track_id, ltrb)

            log = dict(
                track_id = track_id,
                ltrb = ltrb,
                class_id = class_id,
                speed = speed,
                turn_type = turn_type,
                speed_violation = speed_violation,
                wrong_way_violation = wrong_way_violation,
                red_light_violation = red_light_violation
            )

            self.draw_track(frame, log)
            self.submit_violation(violation_frame, log)


        return frame

    def process_video(self):
        """ Process video """
        skip_frame = self.config["skip_frame"]
        video_path = self.config["samples"][self.camera_name]["video_path"]
        output_path = self.config["samples"][self.camera_name]["output_path"]
        frame_size = (1280, 960)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1
        desired_fps = int(fps/skip_frame)
        self.speed_estimator.fps = fps
        self.wrong_way.fps = desired_fps
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        frame_counter = 0
        start_time = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            frame_counter += 1
            if frame_counter % skip_frame != 0: continue


            frame = self.process_frame(frame)

            if frame_counter % desired_fps == 0:
                process_time = time.time() - start_time
                start_time = time.time()
                print(f"Process in a second: {process_time} (current process fps is {desired_fps})")

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

    def __del__(self):
        """Close ThreadPoolExecutor when program ends."""
        self.executor.shutdown(wait=True)
        print("Close all threads!")