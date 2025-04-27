import cv2
import time
import numpy as np
import traceback
from datetime import datetime
from typing import Literal
from concurrent.futures import ThreadPoolExecutor

from .adaptive_frame_skipper import AdaptiveFrameSkipper
from ..engines import VehicleDetector, DeepSORT, RedLightViolationDetector, \
    SpeedEstimator, WrongLaneDrivingDetector, RoadManager
from ..process import AsyncCloudinaryUploader, ViolationManager


class Controller:
    def __init__(self,
                 config,
                 camera_name="1"):
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
        self.video_path = None
        self.cap = None
        self.show_annotations = True
        self.lane_annotate_enabled = False
        self.road_annotate_enabled = False
        self.intersection_annotate_enabled = False
        self.speed_estimation_enabled = True
        self.red_light_detection_enabled = True
        self.wrong_lane_detection_enabled = True

    def init_components(self):
        self.vehicle_detector = VehicleDetector(self.config)
        self.tracker = DeepSORT(self.config)
        self.road_manager = RoadManager(self.config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(
            self.road_manager, self.uploader)
        self.wrong_way = WrongLaneDrivingDetector(
            self.road_manager, self.uploader)
        self.speed_estimator = SpeedEstimator(self.road_manager, self.uploader)

    def draw_track(self, frame, log):
        violation = []

        class_id_log = log["class_id"]
        track_id_log = log["track_id"]
        vehicle_class = self.class_names[class_id_log]

        label = f"{vehicle_class}-{track_id_log}"

        if self.speed_estimation_enabled and log["speed_violation"]:
            violation.append("Speeding")

        if self.red_light_detection_enabled and log["red_light_violation"]:
            violation.append("Red Light Violation")

        if self.wrong_lane_detection_enabled and log["wrong_way_violation"]:
            violation.append("Wrong Way Driving")

        violation_text = "-".join(violation)

        violation_flag = self.violation_manager.is_violated_already(label)

        if not violation_text:
            if violation_flag:
                violation_text = violation_flag

        x1, y1, x2, y2 = map(int, log["ltrb"])
        if violation_flag:
            color = (0, 0, 255)  # Red color in BGR format
        else:
            color = self.colors[log["class_id"]]

        B, G, R = map(int, color)

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

        if violation_flag:
            cv2.rectangle(frame, (x1 - 1, y2),
                          (x1 + len(violation_text) * 12, y2 + 20), (B, G, R), -1)
            cv2.putText(frame, violation_text, (x1 + 5, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def draw_road_structure(self, frame):
        if self.lane_annotate_enabled:
            self.road_manager.draw_lanes(frame)

        if self.road_annotate_enabled:
            self.road_manager.draw_roads(frame)

        if self.intersection_annotate_enabled:
            if self.road_manager.intersection:
                self.road_manager.draw_intersections(frame)

    def process_frame(self, frame):
        """ Process frame

        Args:
            frame (np.array): Frame of video
            frame_position (int, optional): Position of frame in video.
        """

        current_timestamp = time.time()

        boxes = self.vehicle_detector.detect(frame)
        detections = self.tracker.extract_detections(boxes)
        tracks = self.tracker.update_tracks(detections, frame=frame)

        violation_frame = frame.copy()

        if self.red_light_detection_enabled:
            frame = self.rlv_detector.detect_traffic_light_color(frame)

        if self.show_annotations:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id

                ltrb = track.to_ltrb()
                class_id = int(track.get_det_class())
                speed_violation, speed, speed_limit = False, 0, 0
                wrong_way_violation, turn_type = False, "unknown"
                red_light_violation = False

                if self.speed_estimation_enabled:
                    speed_violation, speed, speed_limit = self.speed_estimator.detect_speed(
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
                    speed_limit=speed_limit,
                    turn_type=turn_type,
                    speed_violation=speed_violation,
                    wrong_way_violation=wrong_way_violation,
                    red_light_violation=red_light_violation
                )

                self.handle_violation(log, violation_frame)

                self.draw_track(frame, log)

        return frame

    def init_process_video(self):
        """ Initialize video processing """

        try:
            self.video_path = self.config["samples"][self.camera_name]["video_path"]
            self.update_parameters(self.params) if hasattr(
                self, 'params') else {}

            self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            if not self.cap.isOpened():
                raise Exception(f"Error opening video file: {self.video_path}")

            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.speed_estimator.fps = fps
            self.frame_skipper.target_fps = fps
            self.frame_skipper.fps_timer = time.time()

        except Exception as e:
            print(f"Error initializing video: {str(e)}")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            raise

    def yield_from_video(self):
        try:
            self.init_process_video()
            video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            PAUSED_FPS = 5
            paused_sleep_time = 1.0/PAUSED_FPS

            while True:
                if not self.cap.isOpened():
                    print("Video capture is not opened. Reinitializing...")
                    self.init_process_video()
                    continue

                if self.switch_model_flag:
                    self.vehicle_detector.switch_model(self.new_model)
                    self.switch_model_flag = False

                # Handle pause state
                if self.is_paused:
                    if hasattr(self, 'current_frame') and self.current_frame is not None:
                        # Add pause indicator text

                        if self.show_annotations:
                            paused_frame = self.current_frame.copy()
                            self.draw_road_structure(paused_frame)
                        else:
                            paused_frame = self.frame_origin

                        cv2.putText(paused_frame, "PAUSED", (paused_frame.shape[1]//2 - 100, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                        _, jpeg = cv2.imencode(".jpg", paused_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
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

                process_start_time = time.time()

                ret, frame = self.cap.read()

                if not ret:
                    print("End of video - restarting from beginning")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path)

                    self.frame_skipper.fps_timer = time.time()
                    self.frame_skipper.frame_counter = 0

                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error reading first frame after reset")
                        continue

                self.frame_origin = frame.copy()

                if self.frame_skipper.is_skipable():
                    continue

                try:
                    frame = self.process_frame(frame)

                    if self.frame_skipper.frame_counter >= 5:
                        self.speed_estimator.fps = max(
                            1.0, self.frame_skipper.current_fps)

                    processing_time = time.time() - process_start_time
                    self.frame_skipper.adjust_skip_rate(
                        processing_time, video_fps)
                    self.frame_skipper.update_fps()

                    if self.frame_skipper.frame_counter > 5:
                        self.speed_estimator.fps = self.frame_skipper.current_fps

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

                    self.current_frame = frame.copy()

                    self.draw_road_structure(frame)
                    _, jpeg = cv2.imencode(".jpg", frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    traceback.print_exc()
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Error processing frame", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    _, jpeg = cv2.imencode(".jpg", blank_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.5)

        except Exception as e:
            print(f"Error in video streaming: {str(e)}")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            raise
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()

    def toggle_pause(self):
        """Toggle the pause state of the video."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def switch_model(self,
                     model_type: Literal["yolo11", "rtdetrv2", "faster_rcnn"]):
        """Switch model type"""
        self.new_model = model_type
        self.switch_model_flag = True

    def switch_camera(self, camera_name):
        """Switch camera"""
        try:
            self.new_camera_name = camera_name

            self.camera_name = self.new_camera_name
            self.reinitialize_camera()
            self.init_process_video()
            print(f"Camera switched to {self.camera_name}")

            self.current_frame = self.cap.read()[1]
            self.frame_skipper.fps_timer = time.time()
            self.frame_skipper.frame_counter = 0
            self.frame_skipper.total_skip_frames = 0

            return self.get_system_config()

        except Exception as e:
            print(f"Error switching camera: {e}")
            raise

    def reinitialize_camera(self):
        """Reinitialize components when switching camera"""
        print("Reinitializing components for new camera...")
        self.frame_skipper = AdaptiveFrameSkipper(self.config["frame_skipper"])
        self.road_manager = RoadManager(self.config, self.camera_name)
        self.rlv_detector = RedLightViolationDetector(
            self.road_manager, self.uploader)
        self.wrong_way = WrongLaneDrivingDetector(
            self.road_manager, self.uploader)
        self.speed_estimator = SpeedEstimator(self.road_manager, self.uploader)

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

    def update_speed_limit(self, road_id, speed_limit, lane_id=None):
        """Update speed limit for a road or specific lane"""
        try:
            if road_id == "intersection":
                self.road_config["roads"]["intersection"]["speed_limit"] = speed_limit
            elif lane_id:
                self.road_config["roads"][road_id][lane_id]["speed_limit"] = speed_limit
        except Exception as e:
            print(f"Error updating speed limit: {e}")

    def update_parameters(self, params):
        """Update system parameters based on frontend settings"""
        try:
            self.params = params

            # General settings
            if "general_setting" in params:
                general = params["general_setting"]
                self.vehicle_detector.conf_threshold = general["conf_threshold"]
                self.vehicle_detector.iou_threshold = general["iou_threshold"]
                # Store annotation settings
                self.lane_annotate_enabled = general["lane_annotate_enabled"]
                self.road_annotate_enabled = general["road_annotate_enabled"]
                self.intersection_annotate_enabled = general["intersection_annotate_enabled"]

            # Frame skipper
            if "frame_skipper" in params:
                skipper = params["frame_skipper"]
                self.frame_skipper.target_fps = skipper["target_fps"]
                self.frame_skipper.skip_rate = skipper["skip_rate"]

            # Speed estimation
            if "speed_estimation" in params:
                speed = params["speed_estimation"]
                self.speed_estimation_enabled = speed["enabled"]
                self.speed_estimator.overspeed_buffer = speed["over_speed_buffer"]

                # Update road speed limits
                if "roads" in speed:
                    for road_id, road_data in speed["roads"].items():
                        if road_id == "intersection":
                            self.road_manager.update_speed_limit(
                                road_id, road_data["speed_limit"])
                        else:
                            for lane_id, lane_data in road_data.items():
                                if lane_id.startswith("lane_"):
                                    self.road_manager.update_speed_limit(
                                        road_id, lane_data["speed_limit"], lane_id)

            # Red light violation
            if "red_light_violation" in params:
                self.red_light_detection_enabled = params["red_light_violation"]["enabled"]

            # Wrong way violation
            if "wrong_way_violation" in params:
                wrong_way = params["wrong_way_violation"]
                self.wrong_lane_detection_enabled = wrong_way["enabled"]
                self.wrong_way.angle_threshold = wrong_way["angle_threshold"]
                self.wrong_way.straight_threshold = wrong_way["straight_threshold"]
                self.wrong_way.dot_threshold = wrong_way["dot_threshold"]
                self.wrong_way.tolerance_time = wrong_way["tolerance_time"]

            print("Parameters updated successfully")  # Debug log

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

            speed = int(log["speed"])
            speed_limit = int(log["speed_limit"])
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

            details = "Vehicle driving in wrong direction"

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

    def get_system_config(self):
        """Get current system configuration"""
        try:
            # Get road configuration from road manager
            road_config = self.road_manager.road_config
            road_speed_settings = {}

            # Dynamically build road speed settings
            for road_id, road_data in road_config.items():
                if road_id == "intersection":
                    road_speed_settings[road_id] = {
                        "speed_limit": self.road_manager.get_speed_limit("intersection")
                    }
                    self.intersection_annotate_enabled = True
                else:
                    road_speed_settings[road_id] = {}
                    # Get lanes if they exist
                    lanes = {k: v for k, v in road_data.items()
                             if k.startswith("lane_")}
                    for lane_id in lanes:
                        road_speed_settings[road_id][lane_id] = {
                            "speed_limit": self.road_manager.get_speed_limit(road_id, lane_id)
                        }
                    self.lane_annotate_enabled = True
                    self.road_annotate_enabled = True

            return {
                "general_setting": {
                    "conf_threshold": self.vehicle_detector.conf_threshold,
                    "iou_threshold": self.vehicle_detector.iou_threshold,
                    "lane_annotate_enabled": self.lane_annotate_enabled,
                    "road_annotate_enabled": self.road_annotate_enabled,
                    "intersection_annotate_enabled": self.intersection_annotate_enabled
                },
                "frame_skipper": {
                    "target_fps": self.frame_skipper.target_fps,
                    "skip_rate": self.frame_skipper.skip_rate
                },
                "speed_estimation": {
                    "roads": road_speed_settings,
                    "enabled": self.speed_estimation_enabled,
                    "over_speed_buffer": self.speed_estimator.overspeed_buffer
                },
                "red_light_violation": {
                    "enabled": self.red_light_detection_enabled
                },
                "wrong_way_violation": {
                    "enabled": self.wrong_lane_detection_enabled,
                    "angle_threshold": self.wrong_way.angle_threshold,
                    "straight_threshold": self.wrong_way.straight_threshold,
                    "dot_threshold": self.wrong_way.dot_threshold,
                    "tolerance_time": self.wrong_way.tolerance_time
                }
            }

        except Exception as e:
            print(f"Error getting system config: {e}")
            return None
