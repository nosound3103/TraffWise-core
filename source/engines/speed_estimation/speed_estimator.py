import cv2
import numpy as np
from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, lanes, fps: int=30):
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.fps = fps
        self.lanes = lanes
        self.violations_count = {}

    def _update_coordinates(self, track_id: int, point: np.ndarray):
        """Store coordinates for speed estimation."""
        self.coordinates[track_id].append(point)

    def _calculate_speed(self, track_id: int) -> float:
        """Calculate speed in km/h if there are enough points."""
        if len(self.coordinates[track_id]) < 2:
            return 0.0
        start = self.coordinates[track_id][0]
        end = self.coordinates[track_id][-1]
        distance = np.linalg.norm(np.array(start) - np.array(end))
        time = len(self.coordinates[track_id]) / self.fps
        speed = (distance / time) * 3.6  # Convert m/s to km/h

        return speed

    def _get_lane_index(self, position):
        """Find lane"""
        for idx, lane in enumerate(self.lanes):
            if cv2.pointPolygonTest(lane.source_polygon, position, False) >= 0:
                return idx
        return None
    
    def estimate_speed(self, track_id, bbox, frame):
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        
        speed_violation = False

        lane_index = self._get_lane_index(position)
        if lane_index is None:
            self.coordinates.pop(track_id, None)
            self.violations_count.pop(track_id, None)
            return False, 0.0
                
        transformed_point = self.lanes[lane_index].transform(np.array([[x_center, y_center]]))[0]
        self._update_coordinates(track_id, transformed_point)

        speed = self._calculate_speed(track_id)
        if speed > self.lanes[lane_index].speed_limit:
            self.violations_count[track_id] = self.violations_count.get(track_id, 0) + 1
            if self.violations_count[track_id] > 30:
                        speed_violation = True
        return speed_violation, speed

    def _over_speed_capture(self, speed, frame):
        pass