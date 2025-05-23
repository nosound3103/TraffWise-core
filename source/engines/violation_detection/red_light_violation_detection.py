import cv2
import numpy as np
from collections import OrderedDict


class RedLightViolationDetector:
    def __init__(self, road_manager, uploader, max_track=50):
        self.road_manager = road_manager
        self.max_track = max_track
        self.uploader = uploader
        self.red_1_range = [
            [0, 128, 128],
            [10, 255, 255]
        ]
        self.red_2_range = [
            [170, 128, 128],
            [180, 255, 255]
        ]
        self.orange_range = [
            [5, 120, 70],
            [25, 255, 255]
        ]

        self.car_states = OrderedDict()

    def detect_traffic_light_color(self, frame: np.ndarray):
        """Detect traffic light color."""
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for road in self.road_manager.roads:
            if not road.traffic_light:
                road.traffic_light_status = None
                self.is_red = None
                continue

            traffic_light_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            area = np.array(road.traffic_light, np.int32)
            cv2.fillPoly(traffic_light_mask, [area], 255)

            red_1_mask = cv2.inRange(frame_hsv, np.array(
                self.red_1_range[0]), np.array(self.red_1_range[1]))
            red_2_mask = cv2.inRange(frame_hsv, np.array(
                self.red_2_range[0]), np.array(self.red_2_range[1]))
            red_mask = cv2.bitwise_or(red_1_mask, red_2_mask)

            orange_mask = cv2.inRange(frame_hsv, np.array(
                self.orange_range[0]), np.array(self.orange_range[1]))

            stop_mask = cv2.bitwise_or(red_mask, orange_mask)

            frame_red = cv2.bitwise_and(stop_mask, traffic_light_mask)
            self.is_red = cv2.countNonZero(frame_red) > 0

            road.traffic_light_status = self.is_red

            color = (0, 255, 0) if not road.traffic_light_status else (0, 0, 255)
            status = "GO" if not road.traffic_light_status else "STOP"
            stop_area = np.array(road.stop_area, np.int32)
            if self.is_red:
                cv2.polylines(frame, [stop_area],
                              isClosed=True, color=color, thickness=3)

            cv2.polylines(
                frame, [np.array(road.traffic_light, np.int32)], True, color, 2)
            cv2.putText(
                frame,
                status,
                (int(road.traffic_light[0][0]), int(road.traffic_light[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def detect_red_light_violation(self, track_id, bbox):
        """Detect red light violation"""

        if len(self.car_states) >= self.max_track:
            self.car_states.popitem(last=False)

        if track_id not in self.car_states:
            self.car_states[track_id] = {
                "in_stop_area": False, "violated": False}

        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        road = self.road_manager.get_road(center)

        if road is None or road.traffic_light_status is None:
            is_in_stop_area = False

            if self.is_red is None:
                return False

            is_red_light = self.is_red
        else:
            stop_area = np.array(road.stop_area, np.int32)

            is_in_stop_area = cv2.pointPolygonTest(
                stop_area, center, False) >= 0

            is_red_light = road.traffic_light_status

        if is_in_stop_area:
            self.car_states[track_id]["in_stop_area"] = True

        if self.car_states[track_id]["in_stop_area"] and not is_in_stop_area and is_red_light:
            self.car_states[track_id]["violated"] = True

        return self.car_states[track_id]["violated"]
