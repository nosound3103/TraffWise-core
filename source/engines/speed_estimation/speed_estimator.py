import cv2
import numpy as np
from collections import deque, OrderedDict
import time


class SpeedEstimator:
    def __init__(self, road_manager, uploader, fps: int = 30, max_tracks: int = 50):
        self.road_manager = road_manager
        self.uploader = uploader
        self.fps = fps
        self.max_tracks = max_tracks
        self.coordinates = OrderedDict()
        self.timestamps = OrderedDict()
        self.violations_count = OrderedDict()
        self.max_history_seconds = 3.0
        self.overspeed_buffer = 1

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

        total_distance = 0.0
        points = list(self.coordinates[track_id])
        for i in range(len(points) - 1):
            current_pos = np.array(points[i])
            next_pos = np.array(points[i + 1])
            segment_distance = np.linalg.norm(next_pos - current_pos)
            total_distance += segment_distance

        start_time = self.timestamps[track_id][0]
        end_time = self.timestamps[track_id][-1]
        elapsed_time = end_time - start_time

        try:
            speed = (total_distance / elapsed_time) * 3.6
        except:
            speed = 10

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
            tuple: (speed_violation_flag, speed_in_kmh, speed_limit_in_kmh)
        """
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        speed_violation = False

        if timestamp is None:
            timestamp = time.time()

        lane = self.road_manager.get_lane(position)
        road = self.road_manager.get_road(position)
        intersection = self.road_manager.get_intersection(position)

        if not intersection and (not road or not lane):
            return speed_violation, 0.0, 0.0

        area = road if road else intersection
        flag = "road" if road else "intersection"

        transformed_point = area.transform(np.array([[x_center, y_center]]))[0]

        self._update_coordinates(track_id, transformed_point, timestamp)

        speed = self._calculate_speed(track_id)

        speed_limit = lane.speed_limit \
            if flag == "road" else intersection.speed_limit

        if speed > speed_limit + self.overspeed_buffer:
            self.violations_count[track_id] += 1

            violation_threshold = max(3, int(self.fps))

            if self.violations_count[track_id] > violation_threshold:
                self.violations_count[track_id] = 0
                speed_violation = True

        return (speed_violation, speed, speed_limit)
