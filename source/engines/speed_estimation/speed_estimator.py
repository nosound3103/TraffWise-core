import cv2, numpy as np
from datetime import datetime
from collections import defaultdict, deque, OrderedDict

class SpeedEstimator:
    def __init__(self, lane_manager, uploader, fps: int = 30, max_track: int = 50):
        self.lane_manager = lane_manager
        self.uploader = uploader
        self.fps = fps 
        self.max_track = max_track  
        self.coordinates = OrderedDict()
        self.violations_count = OrderedDict()  

    def _update_coordinates(self, track_id: int, point: np.ndarray):
        """Save vehicle coordinates and limit the number of tracked vehicles"""
        if track_id not in self.coordinates:
            if len(self.coordinates) >= self.max_track:
                self.coordinates.popitem(last=False)  
                self.violations_count.popitem(last=False) 

            self.coordinates[track_id] = deque()  
            self.violations_count[track_id] = 0 

        self.coordinates[track_id].append(point)

        while len(self.coordinates[track_id]) > self.fps:
            self.coordinates[track_id].popleft()

    def _calculate_speed(self, track_id: int) -> float:
        """Calculate speed based on moving coordinates"""
        if len(self.coordinates[track_id]) < 2:
            return 0.0
        safe_fps = max(self.fps, 10)
        start = self.coordinates[track_id][0]
        end = self.coordinates[track_id][-1]
        distance = np.linalg.norm(np.array(start) - np.array(end))
        time = len(self.coordinates[track_id]) / safe_fps  

        speed = (distance / time) * 3.6  
        return speed

    def detect_speed(self, track_id, bbox, frame) -> tuple:
        """Determine vehicle speed and check for violations"""
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        speed_violation = False

        lane = self.lane_manager.get_lane(position)

        if lane is None:
            return speed_violation, 0.0

        transformed_point = lane.transform(np.array([[x_center, y_center]]))[0]
        self._update_coordinates(track_id, transformed_point)

        speed = self._calculate_speed(track_id)

        if speed > lane.speed_limit:
            self.violations_count[track_id] += 1  
            if self.violations_count[track_id] > 30:
                self.violations_count[track_id] = 0
                speed_violation = True

        return speed_violation, speed

    def capture_violation(self, frame, log):
        """Take a photo of the violation and upload it to Cloudinary"""
        x1, y1, x2, y2 = map(int, log["ltrb"])

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
        cv2.putText(frame, f"Overspeed: {int(log['speed'])} km/h", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        date = datetime.now().strftime('%Y-%m-%d')
        folder_path = f"traffic/violations/speed/{date}"
        public_id_prefix = f"{folder_path}/{log['track_id']}"

        exists = self.uploader.file_exists_on_cloudinary(public_id_prefix)

        if not exists:
            timestamp = datetime.now().strftime('%H-%M-%S')
            public_id = f"{public_id_prefix}_{timestamp}"
            self.uploader.upload_violation(frame, public_id, folder_path)
            print("Capture violation!")
