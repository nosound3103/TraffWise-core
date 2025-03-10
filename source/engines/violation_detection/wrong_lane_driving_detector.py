import cv2, yaml
import numpy as np
from datetime import datetime
from shapely import Polygon
from ..utils.lane import Lane

class WrongLaneDrivingDetector:
    """
    This class detects vehicles moving in the wrong direction (wrong way) and determines the movement direction (straight, left, right)  
    based on a combination of the direction vector and the angle difference between the estimated movement direction  
    and the expected lane direction.
    
    Parameters:
    angle_threshold: If the angle difference (angle_diff) exceeds this threshold (in degrees) when the vehicle is moving straight, it is considered a violation.
    straight_threshold: If the angle difference is smaller than this threshold, the vehicle is considered to be moving straight.
    dot_threshold: If the dot product is lower than this threshold, it indicates that the vehicle is moving against the expected direction
    """
    def __init__(self, config, camera_name, uploader, window_size=5, angle_threshold=90, straight_threshold=45, dot_threshold=-0.5):
        self.config = config
        self.annotation_path = self.config["samples"][camera_name]["annotation_path"]
        self.uploader = uploader
        self.lanes = self.create_lane_list()
        self.window_size = window_size
        self.angle_threshold = angle_threshold      
        self.straight_threshold = straight_threshold  
        self.dot_threshold = dot_threshold            
        self.position_histories = {}  # Lưu lịch sử tọa độ bottom center của từng track
        self.intersect = self.compute_intersections()
        self.violations_count = {}


    def create_lane_list(self):
        """
        Read YAML file and create a list of Lane objects.
        :return: List of Lane objects.
        """
        with open(self.annotation_path, 'r') as f:
            data = yaml.safe_load(f)

        lanes = []
        roads = data.get("roads", {})

        for road_key, road_info in roads.items():
            poly_coords = road_info.get("polygons", {}).get("coordinates", None)
            if poly_coords is None or len(poly_coords) < 4:
                continue

            # Sử dụng polygon theo thứ tự như trong file YAML
            source_polygon = np.array(poly_coords, dtype=np.float32)

            ed_coords = road_info.get("expected_direction", {}).get("coordinates", None)
            expected_direction_points = [(tuple(ed_coords[0]), tuple(ed_coords[1]))] \
                if ed_coords and len(ed_coords) >= 2 else None

            target_info = road_info.get("TARGET", {})
            target_width = target_info.get("width", 0)
            target_height = target_info.get("height", 0)

            speed_limit = road_info.get("speed_limit", 60)

            lane_obj = Lane(
                source_polygon=source_polygon,
                target_width=int(target_width),
                target_height=int(target_height),
                expected_direction_points=expected_direction_points,
                speed_limit=int(speed_limit)
            )
            lanes.append(lane_obj)

        return lanes

    
    def compute_intersections(self):
        """
        Computes the intersection of all lane polygons using Shapely.
        If the intersection is a MultiPolygon, it splits it into individual Polygon objects.
        
        :return: self.intersect: A list of shapely Polygon objects representing the intersection regions.
        """
        if not self.lanes:
            intersect = []
            return intersect
        
        shapely_polys = [Polygon(lane.source_polygon) for lane in self.lanes]
        
        inter_poly = shapely_polys[0]
        for poly in shapely_polys[1:]:
            inter_poly = inter_poly.intersection(poly)
            if inter_poly.is_empty:
                intersect = []
                return intersect
        
        if inter_poly.geom_type == "MultiPolygon":
            intersect = list(inter_poly.geoms)
        else:
            intersect = [inter_poly]
        return intersect


    def is_point_in_intersection(self, position):
        """
        Checks whether the given position (x, y) lies in any of the computed intersection regions.
        Uses cv2.pointPolygonTest on the approximated contour (exterior coordinates) of each intersection.
        
        :param position: A tuple (x, y) representing the point to test.
        :return: True if the point is inside (or on the edge) of any intersection region, False otherwise.
        """
        if not self.intersect:
            self.compute_intersections()
        for poly in self.intersect:
            contour = np.array(poly.exterior.coords, dtype=np.int32)
            if cv2.pointPolygonTest(contour, position, False) >= 0:
                return True
        return False
        

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

            If the angle difference < straight_threshold, return "straight".
            If cross > 0, return "right" (vehicle is turning right).
            If cross < 0, return "left" (vehicle is turning left).
            Else, return "Straight"
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
    

    def detect_violation(self, track_id, bbox, frame, speed):
        """
        Determine the vehicle's movement direction and detect wrong way violations:
        - Calculate the object's center from the bounding box (bbox) and store it in history (sliding window).
        - Determine the lane based on the center position, using information from Lane objects.
        - Estimate the average movement direction and calculate the dot product, then compute the angle difference.
        - Use classify_turn to classify the movement direction as "straight", "left", or "right".
        - Display the movement label (straight/left/right).
        - Check intersection.
        - If the vehicle is moving straight but the dot product is less than dot_threshold,
          consider it a wrong way violation, crop the image, and draw a bounding box around the violation.
        Return the turn type (turn_type).
        """
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        position = (x_center, y_center)
        wrong_way_violation = False

        if track_id not in self.position_histories:
            self.position_histories[track_id] = []
        self.position_histories[track_id].append(position)

        if len(self.position_histories[track_id]) > self.window_size:
            self.position_histories[track_id].pop(0)
        
        lane_index = self._get_lane_index(position)

        if lane_index is None:
            self.position_histories.pop(track_id, None)
            self.violations_count.pop(track_id, None)
            return wrong_way_violation, "unknown"

        if len(self.position_histories[track_id]) < 2:
            return wrong_way_violation, "unknown"
        
        direction = self._estimate_direction(self.position_histories[track_id])
        if np.linalg.norm(direction) > 1e-6:
            if speed <= 3:
                turn_type = "unknown"
                return wrong_way_violation, turn_type
            dot_product = np.dot(direction, self.lanes[lane_index].expected_direction)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = np.degrees(np.arccos(dot_product))
            turn_type = self.classify_turn(self.lanes[lane_index].expected_direction, direction)

            # Check intersection
            if self.is_point_in_intersection(position):
                return wrong_way_violation, turn_type

            # Check wrong way
            if dot_product < self.dot_threshold or angle_diff > self.angle_threshold:
                if speed > 3:
                    self.violations_count[track_id] = self.violations_count.get(track_id, 0) + 1
                    if self.violations_count[track_id] > 30:
                        self.violations_count.pop(track_id, None)
                        wrong_way_violation = True
                        # self.capture_violation(track_id, frame, bbox)
            return wrong_way_violation, turn_type
        return wrong_way_violation, "unknown"
    

    def _get_lane_index(self, position):
        """Find lane"""
        for idx, lane in enumerate(self.lanes):
            if cv2.pointPolygonTest(lane.source_polygon, position, False) >= 0:
                return idx
        return None
    

    def capture_violation(self, frame, log):
        """Capture."""
        x1, y1, x2, y2 = map(int, log["ltrb"])  # Lấy tọa độ bbox

        # Draw violation box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Màu đỏ, độ dày 10px
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
    

    def draw_lanes(self, frame):
        for lane in self.lanes:
            pts = lane.source_polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    