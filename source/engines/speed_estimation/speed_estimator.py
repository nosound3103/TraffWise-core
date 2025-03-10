import cv2, asyncio
import numpy as np
from datetime import datetime
from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, lanes, uploader, fps: int=30):
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.fps = fps
        self.lanes = lanes
        self.uploader = uploader
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
    
    def detection_speed(self, track_id, bbox, frame):
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        
        speed_violation = False

        lane_index = self._get_lane_index(position)
        if lane_index is None:
            self.coordinates.pop(track_id, None)
            self.violations_count.pop(track_id, None)
            return speed_violation, 0.0
                
        transformed_point = self.lanes[lane_index].transform(np.array([[x_center, y_center]]))[0]
        self._update_coordinates(track_id, transformed_point)

        speed = self._calculate_speed(track_id)
        if speed > self.lanes[lane_index].speed_limit:
            self.violations_count[track_id] = self.violations_count.get(track_id, 0) + 1
            if self.violations_count[track_id] > 30:
                self.violations_count.pop(track_id, None)
                speed_violation = True
        return speed_violation, speed

    def capture_violation(self, frame, log):
        """Capture."""
        x1, y1, x2, y2 = map(int, log["ltrb"])  # Lấy tọa độ bbox

        # Draw violation box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Màu đỏ, độ dày 10px
        cv2.putText(frame, f"Overspeed: {int(log["speed"])} km/h", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        date = datetime.now().strftime('%Y-%m-%d')
        folder_path = f"traffic/violations/speed/{date}"
        public_id_prefix = f"{folder_path}/{log["track_id"]}"

        exists = self.uploader.file_exists_on_cloudinary(public_id_prefix)

        if not exists:
            timestamp = datetime.now().strftime('%H-%M-%S')
            public_id = f"{public_id_prefix}_{timestamp}"
            self.uploader.upload_violation(frame, public_id, folder_path)
            print("Capture violation!")