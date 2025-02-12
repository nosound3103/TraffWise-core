import cv2
import numpy as np
import yaml
from collections import defaultdict


class RedLightViolationDetector:
    def __init__(self, config, camera_name):
        self.config = config
        self.annotation_path = self.config["samples"][camera_name]["annotation_path"]

        self.read_annotation()

        self.red_range = [
            [0, 100, 100],
            [10, 255, 255]
        ]

        self.green_range = [
            [0, 43, 184],
            [56, 132, 255]
        ]

        self.car_ids_in_stop_area = defaultdict(set)

    def read_annotation(self):
        """ Read annotation file """

        # self.stop_areas = []
        # self.traffic_lights = []

        with open(self.annotation_path, "r") as file:
            self.data = yaml.safe_load(file)

        for road_name, road_info in self.data["roads"].items():
            if road_info['traffic_lights']['coordinates']:
                self.traffic_light = road_info['traffic_lights']['coordinates']

        # for shape in data["shapes"]:
        #     if shape["label"] == "stop_area":
        #         stop_area = shape["points"]
        #         stop_area = np.array(stop_area, np.int32)
        #         self.stop_areas.append(stop_area)
        #     elif shape["label"] == "traffic_light":
        #         points = shape["points"]
        #         points = np.array(points, np.int32)
        #         self.traffic_lights.append(points)

    def check_road_access(self, current_traffic_light_status):
        """
        Determine which road to use based on traffic light status.

        Args:
            current_traffic_light_status (bool): 
                True if traffic light is green, 
                False if red

        Returns:
            tuple: (accessible_road, blocked_road) - Names of the roads that can and cannot be accessed
        """
        # Find which road has traffic lights
        road_with_light = None
        road_without_light = None

        for road_name, road_info in self.data["roads"].items():
            if road_info['traffic_lights']['coordinates']:
                road_with_light = road_name
            else:
                road_without_light = road_name

        if not road_with_light:
            raise ValueError(
                "No road with traffic lights found in the configuration")

        # Determine which road to use based on traffic light status
        if current_traffic_light_status:  # Green light
            return road_with_light
        else:  # Red light
            return road_without_light

    def detect_traffic_light_color(self, frame):
        """ Detect traffic light color

        Args:
            frame(np.array): Frame of video
        """

        traffic_light_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for i in range(len(self.traffic_light)):
            area = np.array(self.traffic_light[i], np.int32)
            cv2.fillPoly(traffic_light_mask, [area], 255)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask = cv2.inRange(frame_hsv, np.array(
            self.red_range[0]), np.array(self.red_range[1]))
        frame_red = cv2.bitwise_and(red_mask, traffic_light_mask)
        is_red = cv2.countNonZero(frame_red) > 0

        self.detected_color = None

        for traffic_light in self.traffic_light:
            traffic_light = np.array(traffic_light, np.int32)
            status = "GO" if not is_red else "STOP"
            color = (0, 255, 0) if not is_red else (0, 0, 255)

            cv2.polylines(
                frame, [traffic_light],
                True, color, 2)
            cv2.putText(frame, status,
                        traffic_light[0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            self.detected_color = is_red

        return frame, self.detected_color

    def detect_red_light_violation(self, frame, tracks):
        """ Detect red light violation

        Args:
            frame(np.array): Frame of video
            boxes(list): List of boxes(Contain ids of vehicles)
        """

        road_access = self.check_road_access(self.detected_color)
        stop_areas = self.data["roads"][road_access]['stop_areas']['coordinates']
        stop_areas = [np.array(stop_area, np.int32)
                      for stop_area in stop_areas]

        for stop_area in stop_areas:
            cv2.polylines(
                frame, [stop_area],
                True, (0, 0, 255), 2)

        violation_logs = []

        for track in tracks:
            violation_flag = None

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = int(track.get_det_class())
            x1, y1, x2, y2 = map(int, ltrb)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            is_in_stop_area = any([
                cv2.pointPolygonTest(stop_area, center, False) >= 0
                for stop_area in stop_areas
            ])

            if track_id not in self.car_ids_in_stop_area[class_id] and is_in_stop_area:
                self.car_ids_in_stop_area[class_id].add(track_id)
            elif track_id not in self.car_ids_in_stop_area[class_id] and not is_in_stop_area:
                violation_flag = False
            else:
                if not is_in_stop_area:
                    if self.detected_color:
                        violation_flag = True
                    else:
                        violation_flag = False
                else:
                    violation_flag = False

            violation = "Red light violation" if violation_flag else "Safe"
            color = (0, 0, 255) if violation_flag else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, violation,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            violation_logs.append({
                "track": track,
                "violation_type": violation_flag
            })

        return frame, violation_logs
