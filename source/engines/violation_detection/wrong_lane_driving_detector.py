import cv2, yaml
import numpy as np
from datetime import datetime
from collections import defaultdict, deque, OrderedDict

class WrongLaneDrivingDetector:
    """
    This class detects vehicles moving in the wrong direction (wrong way) and determines the movement direction (straight, left, right)  
    based on a combination of the direction vector and the angle difference between the estimated movement direction  
    and the expected lane direction.
    """
    def __init__(self, lane_manager, uploader, window_size=5, angle_threshold=90, straight_threshold=30, dot_threshold=-0.5, max_tracks=50):
        self.lane_manager = lane_manager
        self.uploader = uploader
        self.max_tracks = max_tracks
        self.window_size = window_size
        self.angle_threshold = angle_threshold      
        self.straight_threshold = straight_threshold  
        self.dot_threshold = dot_threshold            
        self.position_histories = OrderedDict()  
        self.violations_count = OrderedDict()

    def _estimate_direction(self, points):
        """
        Estimate the average movement direction based on a sliding window of positions.
        """
        if len(points) < 2:
            return np.array([0.0, 0.0])
        diffs = [np.array(points[i+1]) - np.array(points[i]) for i in range(len(points) - 1)]
        avg_diff = np.mean(diffs, axis=0)
        norm = np.linalg.norm(avg_diff)
        return avg_diff / norm if norm > 1e-6 else np.array([0.0, 0.0])

    def classify_turn(self, expected, estimated):
        """
        Compute the cross product and angle between the estimated movement vector and the expected direction vector.
        """
        cross = expected[0] * estimated[1] - expected[1] * estimated[0]
        dot = np.dot(expected, estimated)
        dot = np.clip(dot, -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(dot))
        
        if 0 <= angle_diff <= self.straight_threshold:
            return "straight"
        elif cross > 0:
            return "right"
        elif cross < 0:
            return "left"
        else:
            return "straight"
    
    def _update_position_history(self, track_id: int, position):
        """Save vehicle positions and limit the number of tracked vehicles"""
        if track_id not in self.position_histories:
            if len(self.position_histories) >= self.max_tracks:
                self.position_histories.popitem(last=False)  
                self.violations_count.popitem(last=False)

            self.position_histories[track_id] = deque(maxlen=self.window_size) 
            self.violations_count[track_id] = 0  

        self.position_histories[track_id].append(position)

    def detect_violation(self, track_id, bbox, frame, speed):
        """
        Determine the vehicle's movement direction and detect wrong way violations
        """
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        wrong_way_violation = False

        self._update_position_history(track_id, position)

        lane = self.lane_manager.get_lane(position)
        if lane is None:
            return wrong_way_violation, "unknown"

        if len(self.position_histories[track_id]) < 2:
            return wrong_way_violation, "unknown"
    
        direction = self._estimate_direction(self.position_histories[track_id])
        if np.linalg.norm(direction) > 1e-6:
            if speed <= 3:
                return wrong_way_violation, "unknown"

            dot_product = np.dot(direction, lane.expected_direction)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = np.degrees(np.arccos(dot_product))
            turn_type = self.classify_turn(lane.expected_direction, direction)

            # Check intersection
            if self.lane_manager.is_point_in_intersection(position):
                return wrong_way_violation, turn_type

            # Check wrong way
            if dot_product < self.dot_threshold or angle_diff > self.angle_threshold:
                turn_type = self.classify_turn(lane.expected_direction[::-1], direction)
                if speed > 3:
                    self.violations_count[track_id] += 1
                    if self.violations_count[track_id] > 30:
                        self.violations_count[track_id] = 0
                        wrong_way_violation = True

            return wrong_way_violation, turn_type
        return wrong_way_violation, "unknown"

    def capture_violation(self, frame, log):
        """Take a photo of the violation and upload it to Cloudinary"""
        x1, y1, x2, y2 = map(int, log["ltrb"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
        cv2.putText(frame, "Wrong way", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        date = datetime.now().strftime('%Y-%m-%d')
        folder_path = f"traffic/violations/wrong_way/{date}"
        public_id_prefix = f"{folder_path}/{log["track_id"]}"

        exists = self.uploader.file_exists_on_cloudinary(public_id_prefix)

        if not exists:
            timestamp = datetime.now().strftime('%H-%M-%S')
            public_id = f"{public_id_prefix}_{timestamp}"
            self.uploader.upload_violation(frame, public_id, folder_path)
            print("Capture violation!")
