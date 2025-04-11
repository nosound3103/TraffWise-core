import cv2
import time
import numpy as np
import os
from datetime import datetime
from typing import Literal, Dict
from concurrent.futures import ThreadPoolExecutor

from .adaptive_frame_skipper import AdaptiveFrameSkipper
from ..engines import VehicleDetector, DeepSORT, RedLightViolationDetector, \
    SpeedEstimator, WrongLaneDrivingDetector, LaneManager
from ..process import AsyncCloudinaryUploader, ViolationManager


class Controller:
    def __init__(self,
                 config,
                 camera_name:
                     Literal[
                         "1",
                         "2"] = "1"):
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
        self.violation_manager = ViolationManager(self.class_names)
        self.executor = ThreadPoolExecutor(max_workers=3)

        self.init_components()

        self.switch_model_flag = False
        self.switch_camera_flag = False
        self.new_model = None
        self.new_camera_name = None
        self.current_frame = None
        self.is_paused = False
        self.current_video_position = 0
        self.video_path = None
        self.current_cap = None
        self.show_annotations = True
        self.speed_estimation_enabled = True
        self.red_light_detection_enabled = True
        self.wrong_lane_detection_enabled = True

    def init_components(self):
        self.vehicle_detector = VehicleDetector(self.config["models"])
        self.tracker = DeepSORT(self.config)
        self.lane_manager = LaneManager(self.config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(
            self.lane_manager, self.uploader)
        self.wrong_way = WrongLaneDrivingDetector(
            self.lane_manager, self.uploader)
        self.speed_estimator = SpeedEstimator(self.lane_manager, self.uploader)

    def draw_track(self, frame, log):
        violation = []

        if self.speed_estimation_enabled and log["speed_violation"]:
            violation.append("Overspeeding")

        if self.red_light_detection_enabled and log["red_light_violation"]:
            violation.append("RLV")

        if self.wrong_lane_detection_enabled and log["wrong_way_violation"]:
            violation.append("Wrong way")

        violation_text = "-".join(violation)

        x1, y1, x2, y2 = map(int, log["ltrb"])
        color = self.colors[log["class_id"]]
        B, G, R = map(int, color)

        class_id_log = log["class_id"]
        track_id_log = log["track_id"]

        label = f"{self.class_names[class_id_log]}-{track_id_log}"

        if self.speed_estimation_enabled:
            speed_log = int(log["speed"])
            label += f"-{speed_log} km/h"

        if self.wrong_lane_detection_enabled:
            label += f"-{log['turn_type']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)

        cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(label) * 12, y1), (B, G, R), -1)

        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if violation_text:
            cv2.rectangle(frame, (x1 - 1, y2),
                          (x1 + len(violation_text) * 12, y2 + 20), (B, G, R), -1)
            cv2.putText(frame, violation_text, (x1 + 5, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def process_frame(self, frame):
        """ Process frame

        Args:
            frame (np.array): Frame of video
            frame_position (int, optional): Position of frame in video.
        """

        if self.switch_model_flag:
            self.vehicle_detector.switch_model(self.new_model)
            self.switch_model_flag = False

        current_timestamp = time.time()

        boxes = self.vehicle_detector.detect(frame)
        detections = self.tracker.extract_detections(boxes)
        tracks = self.tracker.update_tracks(detections, frame=frame)

        violation_frame = frame.copy()

        if hasattr(self, 'red_light_detection_enabled') \
                and self.red_light_detection_enabled:
            frame = self.rlv_detector.detect_traffic_light_color(frame)

        if self.show_annotations:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = int(track.get_det_class())

                speed_violation, speed = False, 0
                wrong_way_violation, turn_type = False, "unknown"
                red_light_violation = False

                if self.speed_estimation_enabled:
                    speed_violation, speed = self.speed_estimator.detect_speed(
                        track_id, ltrb, current_timestamp)

                if self.wrong_lane_detection_enabled:
                    wrong_way_violation, turn_type = self.wrong_way.detect_violation(
                        track_id, ltrb, speed)

                if self.red_light_detection_enabled:
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

                self.handle_violation(log, violation_frame)

                self.draw_track(frame, log)

        self.current_frame = frame
        return frame

    def init_process_video(self):
        """ Initialize video processing """
        self.video_path = self.config["samples"][self.camera_name]["video_path"]
        output_path = self.config["samples"][self.camera_name]["output_path"]
        self.update_parameters(self.params) if hasattr(self, 'params') else {}

        frame_size = (1280, 960)
        cap = cv2.VideoCapture(self.video_path)
        self.current_cap = cap

        if self.current_video_position > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_video_position)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.speed_estimator.fps = fps
        self.frame_skipper.target_fps = fps
        self.frame_skipper.fps_timer = time.time()
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        return cap, out

    def process_video(self):
        """ Process video """
        cap, out = self.init_process_video()
        frame_size = (1280, 960)
        count = 1

        while True:
            if self.switch_camera_flag:
                cap.release()
                out.release()
                self.camera_name = self.new_camera_name
                self.reinitialize_camera()
                cap, out = self.init_process_video()
                self.switch_camera_flag = False

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

            fps_text = f"FPS: {self.frame_skipper.current_fps:.1f}"
            skip_text = f"Skip: {self.frame_skipper.total_skip_frames} frames"
            proc_text = f"Proc time: {processing_time*1000:.1f}ms"
            model_text = f"Model: {self.vehicle_detector.model_type}"
            camera_text = f"Camera: {self.camera_name}"

            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, skip_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, proc_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, model_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, camera_text, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            resized_frame = cv2.resize(frame, frame_size)

            out.write(resized_frame)

            cv2.imshow("Video Frame", resized_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)

            count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def yield_from_video(self):
        cap, _ = self.init_process_video()
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))

        PAUSED_FPS = 5
        paused_sleep_time = 1.0/PAUSED_FPS

        while True:
            if self.switch_camera_flag:
                if self.current_cap:
                    self.current_video_position = int(
                        self.current_cap.get(cv2.CAP_PROP_POS_FRAMES))

                cap.release()
                self.camera_name = self.new_camera_name
                self.reinitialize_camera()
                cap, _ = self.init_process_video()
                self.current_cap = cap
                self.switch_camera_flag = False
                print(f"Camera switched to {self.camera_name}")

                self.frame_skipper.fps_timer = time.time()
                self.frame_skipper.frame_counter = 0
                self.frame_skipper.total_skip_frames = 0

            # Handle pause state
            if self.is_paused:
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    # Add pause indicator text
                    paused_frame = self.current_frame.copy()
                    cv2.putText(paused_frame, "PAUSED", (paused_frame.shape[1]//2 - 100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                    _, jpeg = cv2.imencode(".jpg", paused_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    # Use much lower frame rate when paused to reduce resource usage
                    time.sleep(paused_sleep_time)
                    continue
                else:
                    # If no frame is available, show pause message
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Paused", (280, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, jpeg = cv2.imencode(".jpg", blank_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(paused_sleep_time)
                    continue

            # Normal video processing when not paused
            process_start_time = time.time()  # Track when we start processing

            ret, frame = cap.read()

            # Get current position
            current_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.current_video_position = current_position

            if not ret:
                print("End of video or error reading frame.")
                # Reset to beginning of video
                cap.release()
                cap, _ = self.init_process_video()
                self.current_cap = cap
                continue

            # Check if frame should be skipped based on our adaptive pattern
            if self.frame_skipper.is_skipable():
                continue  # Skip this frame and get next one

            try:
                # Process the frame with timestamp
                frame = self.process_frame(frame)

                # Update the speed estimator's FPS to match actual processing rate
                if self.frame_skipper.frame_counter >= 5:
                    # Adjust the speed estimator's FPS to match the actual frame rate
                    self.speed_estimator.fps = max(
                        1.0, self.frame_skipper.current_fps)

                # Calculate processing time and adjust skip rate
                processing_time = time.time() - process_start_time
                self.frame_skipper.adjust_skip_rate(processing_time, video_fps)
                self.frame_skipper.update_fps()

                if self.frame_skipper.frame_counter > 5:
                    self.speed_estimator.fps = self.frame_skipper.current_fps

                # Display stats on frame
                fps_text = f"FPS: {self.frame_skipper.current_fps:.1f}"
                skip_text = f"Skip: {self.frame_skipper.total_skip_frames} frames"
                proc_text = f"Proc time: {processing_time*1000:.1f}ms"
                model_text = f"Model: {self.vehicle_detector.model_type}"
                camera_text = f"Camera: {self.camera_name}"

                # if hasattr(self.frame_skipper, 'skip_pattern') and self.frame_skipper.skip_pattern:
                #     pattern = ''.join(
                #         map(str, self.frame_skipper.skip_pattern))
                #     pattern_text = f"Pattern: {pattern}"
                #     cv2.putText(frame, pattern_text, (10, 180),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, skip_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, proc_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, model_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, camera_text, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                _, jpeg = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except Exception as e:
                print(f"Error processing frame: {e}")
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Error processing frame", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, jpeg = cv2.imencode(".jpg", blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.5)

    def toggle_pause(self):
        """Toggle the pause state of the video."""
        if self.is_paused:
            self.frame_skipper.fps_timer = time.time()
            self.frame_skipper.frame_counter = 0
            self.frame_skipper.total_skip_frames = 0

        self.is_paused = not self.is_paused
        return self.is_paused

    def switch_model(self,
                     model_type: Literal["yolo11", "rtdetrv2", "faster_rcnn"]):
        """Switch model type"""
        self.new_model = model_type
        self.switch_model_flag = True

    def switch_camera(self, camera_name: Literal["1", "2"]):
        """Switch camera"""
        self.new_camera_name = camera_name
        self.switch_camera_flag = True

    def reinitialize_camera(self):
        """Reinitialize components when switching camera"""
        print("Reinitializing components for new camera...")
        self.frame_skipper = AdaptiveFrameSkipper(self.config["frame_skipper"])
        self.lane_manager = LaneManager(self.config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(
            self.lane_manager, self.uploader)
        self.wrong_way = WrongLaneDrivingDetector(
            self.lane_manager, self.uploader)
        self.speed_estimator = SpeedEstimator(self.lane_manager, self.uploader)
        self.tracker = DeepSORT(self.config)

    def toggle_annotations(self, show_annotations):
        """
        Toggle whether to show or hide annotations on the video feed.
        Processing continues in the background either way.

        Args:
            show_annotations (bool): Whether to show annotations

        Returns:
            dict: Contains the current state of annotations visibility
        """
        self.show_annotations = show_annotations
        return {"show_annotations": self.show_annotations}

    def __del__(self):
        """Close ThreadPoolExecutor when program ends."""
        self.executor.shutdown(wait=True)
        print("Close all threads!")

    def reset_state(self, camera_id=None):
        """
        Reset the controller to its initial state.
        Optionally switch to a specific camera.
        """
        self.switch_model("yolo11")

        if camera_id:
            self.switch_camera(camera_id)
        else:
            self.switch_camera("1")

        self.frame_skipper.fps_timer = time.time()
        self.frame_skipper.frame_counter = 0
        self.frame_skipper.total_skip_frames = 0

        return {"status": "success", "message": "Controller reset to initial state"}

    def get_current_frame(self):
        """
        Returns the current frame from the video feed.
        This frame will include any detection boxes if a model is active.

        Returns:
            numpy.ndarray: The current frame or None if no frame is available
        """
        try:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                return self.current_frame

            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "No frame available", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return blank_frame
        except Exception as e:
            print(f"Error getting current frame: {str(e)}")
            return None

    def update_parameters(self, params):
        """
        Update system parameters based on frontend settings

        Args:
            params (dict): Dictionary of parameters from frontend
        """

        try:

            self.params = params

            # Update parameters with proper type checking
            if "speedEstimationEnabled" in params:
                self.speed_estimation_enabled = bool(
                    params["speedEstimationEnabled"])
                print(
                    f"Speed estimation enabled: {self.speed_estimation_enabled}")

            if "redLightDetectionEnabled" in params:
                self.red_light_detection_enabled = bool(
                    params["redLightDetectionEnabled"])
                print(
                    f"Red light detection enabled: {self.red_light_detection_enabled}")

            if "wrongLaneDetectionEnabled" in params:
                self.wrong_lane_detection_enabled = bool(
                    params["wrongLaneDetectionEnabled"])
                print(
                    f"Wrong lane detection enabled: {self.wrong_lane_detection_enabled}")
            if "confidenceThreshold" in params:
                self.config["conf_threshold"] = params["confidenceThreshold"]
                self.tracker.conf_threshold = params["confidenceThreshold"]

            if "nmsThreshold" in params:
                self.config["nms_threshold"] = params["nmsThreshold"]

            if "maxAge" in params:
                self.tracker = DeepSORT(self.config, max_age=params["maxAge"])

            if "speedEstimationEnabled" in params:
                self.speed_estimation_enabled = params["speedEstimationEnabled"]

            if "speedLimit" in params:
                for lane in self.lane_manager.lanes:
                    lane.speed_limit = params["speedLimit"]
                print(params["speedLimit"])

            if "overspeedBuffer" in params:
                self.speed_estimator.overspeed_buffer = params["overspeedBuffer"]

            if "maxHistorySeconds" in params:
                self.speed_estimator.max_history_seconds = params["maxHistorySeconds"]

            if "redLightDetectionEnabled" in params:
                self.red_light_detection_enabled = params["redLightDetectionEnabled"]

            if "maxTrackRLV" in params:
                self.rlv_detector = RedLightViolationDetector(
                    self.lane_manager, self.uploader, max_track=params["maxTrackRLV"])

            if "wrongLaneDetectionEnabled" in params:
                self.wrong_lane_detection_enabled = params["wrongLaneDetectionEnabled"]

            if "angleThreshold" in params \
                    or "straightThreshold" in params \
                    or "dotThreshold" in params \
                    or "toleranceTime" in params:
                self.wrong_way = WrongLaneDrivingDetector(
                    self.lane_manager,
                    self.uploader,
                    angle_threshold=params.get("angleThreshold", 90),
                    straight_threshold=params.get("straightThreshold", 30),
                    dot_threshold=params.get("dotThreshold", -0.5),
                    fps=self.wrong_way.fps
                )
                if "toleranceTime" in params:
                    self.wrong_way.fps = self.speed_estimator.fps

        except Exception as e:
            print(f"Error in update_parameters: {str(e)}")
            raise

    def handle_violation(self, log, frame):
        """
        Handle all types of violations in one central place using the log data

        Args:
            log (dict): The tracking and violation data dictionary
            frame (np.ndarray): The current video frame for capturing evidence
        """

        # Handle speed violation
        if self.speed_estimation_enabled \
                and log["speed_violation"]:
            lane = self.lane_manager.get_lane(
                ((log["ltrb"][0] + log["ltrb"][2]) / 2,
                 (log["ltrb"][1] + log["ltrb"][3]) / 2)
            )

            if lane:
                speed_limit = lane.speed_limit
                speed = int(log["speed"])
                details = f"{speed} km/h (Limit: {speed_limit} km/h)"

                image_url = self.capture_violation(frame.copy(), log, "speed")

                self.violation_manager.add_violation(
                    log=log,
                    violation_type="speed",
                    location=f"Camera {self.camera_name}",
                    details=details,
                    image_url=image_url
                )

        # Handle red light violation
        if self.red_light_detection_enabled \
                and log["red_light_violation"]:
            details = "Crossed while light was red"

            image_url = self.capture_violation(frame.copy(), log, "rlv")

            self.violation_manager.add_violation(
                log=log,
                violation_type="rlv",
                location=f"Camera {self.camera_name}",
                details=details,
                image_url=image_url
            )

        # Handle wrong way violation
        if self.wrong_lane_detection_enabled \
                and log["wrong_way_violation"]:

            details = f"Vehicle driving in wrong direction)"

            image_url = self.capture_violation(frame.copy(), log, "wrong_way")

            self.violation_manager.add_violation(
                log=log,
                violation_type="wrong_way",
                location=f"Camera {self.camera_name}",
                details=details,
                image_url=image_url
            )

    def capture_violation(self, frame, log, violation_type):
        """
        Capture a violation image and upload it to Cloudinary asynchronously

        Args:
            frame (np.ndarray): The frame to capture
            log (dict): The tracking and violation data
            violation_type (str): Type of violation (speed, rlv, wrong_way)

        Returns:
            str: URL placeholder that will be updated later with the real URL
        """
        try:
            track_id = log["track_id"]
            timestamp = datetime.now().strftime('%H-%M-%S')

            violation_frame = frame.copy()
            violation_log = log.copy()

            placeholder_url = f"pending_upload_{violation_type}_{track_id}_{timestamp}"

            self.executor.submit(
                self._process_violation_image,
                violation_frame,
                violation_log,
                violation_type,
                placeholder_url
            )

            return placeholder_url

        except Exception as e:
            print(f"Error in capture_violation: {e}")
            return None

    def _process_violation_image(self, frame, log, violation_type, placeholder_url):
        """Background worker to process and upload the violation image"""
        try:
            try:
                x1, y1, x2, y2 = map(int, log["ltrb"])
                vehicle_class = self.class_names[log["class_id"]]
                speed = int(log["speed"])
                turn_type = log["turn_type"]
                track_id = log["track_id"]
            except (ValueError, KeyError, TypeError) as e:
                print(f"Error extracting violation data: {e}")
                if "ltrb" not in log:
                    return None
                x1, y1, x2, y2 = map(int, log["ltrb"])
                vehicle_class = "unknown"
                speed = 0
                turn_type = "unknown"
                track_id = log.get("track_id", 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)

            if violation_type == "speed":
                text = f"Overspeed: {speed} km/h"
            elif violation_type == "rlv":
                text = "Red Light Violation"
            elif violation_type == "wrong_way":
                text = f"Wrong Way ({turn_type})"
            else:
                text = "Violation"

            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"Camera: {self.camera_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Vehicle: {vehicle_class}-{track_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            date = datetime.now().strftime('%Y-%m-%d')
            folder_path = f"traffic_violations/{violation_type}/{date}"
            public_id_prefix = f"{folder_path}/{track_id}"

            exists = False
            if hasattr(self.uploader, 'file_exists_on_cloudinary'):
                try:
                    exists = self.uploader.file_exists_on_cloudinary(
                        public_id_prefix)
                except Exception as e:
                    print(f"Error checking if file exists: {e}")

            if not exists:
                timestamp = datetime.now().strftime('%H-%M-%S')
                public_id = f"{public_id_prefix}_{timestamp}"
                result = self.uploader.upload_violation(
                    frame, public_id, folder_path)
                print(f"Captured {violation_type} violation!")

                if result and 'secure_url' in result:
                    real_url = result['secure_url']

                    for violation in self.violation_manager.violations:
                        if violation["evidence"] == placeholder_url:
                            violation["evidence"] = real_url
                            print(f"Updated violation image URL: {real_url}")
                            break

                    return real_url

        except Exception as e:
            print(f"Error processing violation image: {e}")
            return None
