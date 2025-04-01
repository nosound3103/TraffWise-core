import cv2
import time
import numpy as np
from typing import Literal, Dict
from concurrent.futures import ThreadPoolExecutor

from .adaptive_frame_skipper import AdaptiveFrameSkipper
from ..engines import VehicleDetector, DeepSORT, RedLightViolationDetector, \
    SpeedEstimator, WrongLaneDrivingDetector, LaneManager
from ..process import AsyncCloudinaryUploader


class Controller:
    def __init__(self,
                 config,
                 camera_name:
                     Literal[
                         "llgt_fpt",
                         "traffic_vn",
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

        self.vehicle_detector = VehicleDetector(config["models"])
        self.tracker = DeepSORT(config)
        self.lane_manager = LaneManager(config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(
            self.lane_manager, self.uploader)
        self.wrong_way = WrongLaneDrivingDetector(
            self.lane_manager, self.uploader)
        self.speed_estimator = SpeedEstimator(self.lane_manager, self.uploader)

    def draw_track(self, frame, log):
        violation = []

        if log["speed_violation"]:
            violation.append("Overspeeding")
        if log["red_light_violation"]:
            violation.append("RLV")
        if log["wrong_way_violation"]:
            violation.append("Wrong way")

        violation_text = "-".join(violation)

        x1, y1, x2, y2 = map(int, log["ltrb"])
        color = self.colors[log["class_id"]]
        B, G, R = map(int, color)

        class_id_log = log["class_id"]
        track_id_log = log["track_id"]
        speed_log = int(log["speed"])

        label = f"{self.class_names[class_id_log]}-{track_id_log}-{int(speed_log)} km/h-{log['turn_type']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(label) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(violation_text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, violation_text, (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def submit_violation(self, frame, log):
        if log["speed_violation"]:
            print(f"Track_id: {log['track_id']} got over speed violation!")
            frame_copy = frame.copy()
            future = self.executor.submit(
                self.speed_estimator.capture_violation, frame_copy, log)
            # future.add_done_callback(lambda f: print(f"Done process!"))

        if log["red_light_violation"]:
            print(f"Track_id: {log['track_id']} got red light violation!")
            frame_copy = frame.copy()
            future = self.executor.submit(
                self.rlv_detector.capture_violation, frame_copy, log)
            # future.add_done_callback(lambda f: print(f"Done process!"))

        if log["wrong_way_violation"]:
            print(f"Track_id: {log['track_id']} got wrong way violation!")
            frame_copy = frame.copy()
            future = self.executor.submit(
                self.wrong_way.capture_violation, frame_copy, log)
            # future.add_done_callback(lambda f: print(f"Done process!"))

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
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = int(track.get_det_class())
            # Speed Estimate
            speed_violation, speed = self.speed_estimator.detect_speed(
                track_id, ltrb, frame)
            # Wrong way driving detection
            wrong_way_violation, turn_type = self.wrong_way.detect_violation(
                track_id, ltrb, frame, speed)
            # Red light detection
            red_light_violation = self.rlv_detector.detect_red_light_violation(
                track_id, ltrb)

            log = dict(
                track_id=track_id,
                ltrb=ltrb,
                class_id=class_id,
                speed=speed,
                turn_type=turn_type,
                speed_violation=speed_violation,
                wrong_way_violation=wrong_way_violation,
                red_light_violation=red_light_violation
            )

            self.draw_track(frame, log)
            self.submit_violation(violation_frame, log)

        return frame

    def process_video(self):
        """ Process video """
        video_path = self.config["samples"][self.camera_name]["video_path"]
        output_path = self.config["samples"][self.camera_name]["output_path"]
        frame_size = (1280, 960)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.speed_estimator.fps = fps
        self.frame_skipper.target_fps = fps
        self.frame_skipper.fps_timer = time.time()
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while True:

            ret, frame = cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            if self.frame_skipper.is_skipable():
                self.frame_skipper.total_skip_frames += 1
                self.frame_skipper.update_fps()
                continue

            frame = self.process_frame(frame)

            processing_time = time.time() - self.frame_skipper.fps_timer
            self.frame_skipper.adjust_skip_rate(processing_time)
            self.frame_skipper.update_fps()

            # if self.frame_skipper.frame_counter > 5:
            #     self.speed_estimator.fps = self.frame_skipper.current_fps
            #     continue

            fps_text = f"FPS: {self.frame_skipper.current_fps:.1f}"
            skip_text = f"Skip: {self.frame_skipper.total_skip_frames} frames"
            proc_text = f"Proc time: {processing_time*1000:.1f}ms"

            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, skip_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, proc_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
