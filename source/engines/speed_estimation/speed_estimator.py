import cv2
import numpy as np
from collections import deque, OrderedDict
import time


class SpeedEstimator:
    def __init__(self, lane_manager, uploader, fps: int = 30, max_tracks: int = 50):
        self.lane_manager = lane_manager
        self.uploader = uploader
        self.fps = fps
        self.max_tracks = max_tracks
        self.coordinates = OrderedDict()
        self.timestamps = OrderedDict()
        self.violations_count = OrderedDict()
        self.max_history_seconds = 3.0

    def _update_coordinates(self, track_id: int, point: np.ndarray, timestamp: float):
        """
        Save vehicle coordinates with timestamps to handle frame skipping

        Args:
            track_id: The vehicle's tracking ID
            point: The transformed position coordinates
            timestamp: Current timestamp in seconds
        """
        if track_id not in self.coordinates:
            if len(self.coordinates) >= self.max_tracks:
                self.coordinates.popitem(last=False)
                self.timestamps.popitem(last=False)
                self.violations_count.popitem(last=False)

            self.coordinates[track_id] = deque()
            self.timestamps[track_id] = deque()
            self.violations_count[track_id] = 0

        self.coordinates[track_id].append(point)
        self.timestamps[track_id].append(timestamp)

        # Limit history by time rather than frame count to handle frame skipping
        while len(self.timestamps[track_id]) > 1 and timestamp - self.timestamps[track_id][0] > self.max_history_seconds:
            self.coordinates[track_id].popleft()
            self.timestamps[track_id].popleft()

    def _calculate_speed(self, track_id: int) -> float:
        """
        Calculate speed based on actual time elapsed rather than frame count
        to handle frame skipping correctly
        """
        if len(self.coordinates[track_id]) < 2 or len(self.timestamps[track_id]) < 2:
            return 0.0

        # Use first and last position with their corresponding timestamps
        start_pos = self.coordinates[track_id][0]
        end_pos = self.coordinates[track_id][-1]
        start_time = self.timestamps[track_id][0]
        end_time = self.timestamps[track_id][-1]

        # Calculate distance in transformed space
        distance = np.linalg.norm(np.array(start_pos) - np.array(end_pos))

        # Calculate actual elapsed time
        elapsed_time = end_time - start_time

        # Avoid division by zero or unrealistic time values
        if elapsed_time < 0.01:
            return 0.0

        # Convert to km/h (distance in arbitrary units * calibration factor)
        speed = (distance / elapsed_time) * 3.6
        return speed

    def detect_speed(self, track_id, bbox, timestamp=None) -> tuple:
        """
        Determine vehicle speed and check for violations, accounting for frame skipping

        Args:
            track_id: Vehicle tracking ID
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame: Current video frame
            timestamp: Current time in seconds (default: current time)

        Returns:
            tuple: (speed_violation_flag, speed_in_kmh)
        """
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        speed_violation = False

        if timestamp is None:
            timestamp = time.time()

        lane = self.lane_manager.get_lane(position)

        if lane is None:
            return speed_violation, 0.0

        transformed_point = lane.transform(np.array([[x_center, y_center]]))[0]

        self._update_coordinates(track_id, transformed_point, timestamp)

        speed = self._calculate_speed(track_id)

        if speed > lane.speed_limit:
            self.violations_count[track_id] += 1

            violation_threshold = max(3, int(self.fps))

            if self.violations_count[track_id] > violation_threshold:
                self.violations_count[track_id] = 0
                speed_violation = True

        return speed_violation, speed
