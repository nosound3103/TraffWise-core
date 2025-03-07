import cv2
import time
import numpy as np
from typing import Literal, Dict

from .adaptive_frame_skipper import AdaptiveFrameSkipper
from ..engines import VehicleDetector, DeepSORT, \
    RedLightViolationDetector, SpeedEstimator, WrongLaneDrivingDetector
from ..process import AsyncCloudinaryUploader


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
        self.class_names = list(config["labels"].keys())
        self.colors = [
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (192, 192, 192)
        ]
        self.camera_name = camera_name
        self.frame_skipper = AdaptiveFrameSkipper(config["frame_skipper"])
        self.uploader = AsyncCloudinaryUploader()

        self.vehicle_detector = VehicleDetector(config["models"])
        self.tracker = DeepSORT(
            config, self.camera_name)
        # self.rlv_detector = RedLightViolationDetector(
        #     config, self.camera_name)
        self.wrong_way = WrongLaneDrivingDetector(config, self.camera_name)
        self.speed_estimator = SpeedEstimator(self.wrong_way.lanes)
    def draw_track(self, track_id, ltrb, class_id, frame, log):
        x1, y1, x2, y2 = map(int, ltrb)
        color = self.colors[class_id]
        B, G, R = map(int, color)

        # if violation_flag:
        #     self.uploader.upload_violation(frame)

        label = f"{self.class_names[class_id]}-{track_id}-{int(log["speed"])}-km/h"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.putText(frame, log["turn_type"], (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # if log["speed_violation"]:
            # pass
        # if log["red_light_violation"]:
            # pass
        if log["wrong_way_violation"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
            cv2.putText(frame, "Wrong_way", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def process_frame(self, frame):
        """ Process frame

        Args:
            frame (np.array): Frame of video
        """
        boxes = self.vehicle_detector.detect(frame)

        detections = self.tracker.extract_detections(boxes)
        tracks = self.tracker.update_tracks(
            detections, frame=frame)
        # frame = self.rlv_detector.detect_traffic_light_color(
        #     frame)
        
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = int(track.get_det_class())
            # Speed Estimate
            speed_violation, speed = self.speed_estimator.estimate_speed(track_id, ltrb, frame)
            # Wrong way driving detection
            wrong_way_violation, turn_type = self.wrong_way.detect_violation(track_id, ltrb, frame, speed)
            # Red light detection
            # red_light_violation = self.rlv_detector.detect_red_light_violation(track_id, bbox, frame)

            log = dict(
                speed = speed,
                turn_type = turn_type,
                # speed_violation = speed_violation,
                wrong_way_violation = wrong_way_violation,
                # red_light_violation = red_light_violation
            )
            self.draw_track(track_id, ltrb, class_id, frame, log)


        return frame

    def process_video(self):
        """ Process video """
        video_path = self.config["samples"][self.camera_name]["video_path"]
        output_path = self.config["samples"][self.camera_name]["output_path"]
        frame_size = (1280, 960)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.speed_estimator.fps = fps
        self.frame_skipper.target_fps = fps
        self.frame_skipper.fps_timer = time.time()
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while True:

            ret, frame = cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            # if self.frame_skipper.is_skipable():
            #     self.frame_skipper.total_skip_frames += 1
            #     continue

            frame = self.process_frame(frame)

            # processing_time = time.time() - self.frame_skipper.fps_timer
            # self.frame_skipper.adjust_skip_rate(processing_time)
            # self.frame_skipper.update_fps()

            # if self.frame_skipper.frame_counter > 5:
            #     # self.speed_estimator.fps = self.frame_skipper.current_fps
            #     pass

            # fps_text = f"FPS: {self.frame_skipper.current_fps:.1f}"
            # skip_text = f"Skip: {self.frame_skipper.total_skip_frames} frames"
            # proc_text = f"Proc time: {processing_time*1000:.1f}ms"

            # cv2.putText(frame, fps_text, (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, skip_text, (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, proc_text, (10, 90),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
