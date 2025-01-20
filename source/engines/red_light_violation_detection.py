import cv2
import numpy as np


class RedLightViolationDetector:
    def __init__(self, config):
        self.config = config
        self.traffic_light_area = self.config['traffic_light_area']
        self.stop_areas = self.config['stop_area']

        self.red_range = [
            [0, 100, 100],
            [10, 255, 255]
        ]

        self.green_range = [
            [0, 43, 184],
            [56, 132, 255]
        ]

    def detect_traffic_light_color(self, frame):
        """ Detect traffic light color

        Args:
            frame (np.array): Frame of video
        """

        traffic_light_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(traffic_light_mask, [self.traffic_light_area], 255)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask = cv2.inRange(frame_hsv, np.array(
            self.red_range[0]), np.array(self.red_range[1]))
        frame_red = cv2.bitwise_and(frame, red_mask)
        is_red = cv2.countNonZero(frame_red) > 0

        self.detected_color = None

        if is_red:
            cv2.polylines(
                frame, [np.array(self.traffic_light_area, np.int32)],
                True, (0, 0, 255), 2)
            cv2.putText(frame, "STOP",
                        (self.traffic_light_area[0][0],
                         self.traffic_light_area[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.detected_color = "red"
        else:
            cv2.polylines(
                frame, [np.array(self.traffic_light_area, np.int32)],
                True, (0, 255, 0), 2)
            cv2.putText(frame, "GO", (self.traffic_light_area[0][0],
                                      self.traffic_light_area[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.detected_color = "green"

        return frame, self.detected_color

    def detect_red_light_violation(self, frame, boxes):
        """ Detect red light violation

        Args:
            frame (np.array): Frame of video
            boxes (list): List of boxes (Contain ids of vehicles)
        """

        for stop_area in self.stop_areas:
            cv2.polylines(
                frame, [np.array(stop_area, np.int32)],
                True, (0, 0, 255), 2)

        violation_logs = []

        for box in boxes:
            _, _, x1, y1, x2, y2 = box
            center = [(x1 + x2) // 2, (y1 + y2) // 2]

            is_in_stop_area = any([
                cv2.pointPolygonTest(stop_area, tuple(center), False) >= 0
                for stop_area in self.stop_areas
            ])

            if not is_in_stop_area:
                if self.detected_color == "red":
                    violation_flag = "Red Light Violation"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, violation_flag,
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

                elif self.detected_color == "green":
                    violation_flag = "Safe"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, violation_flag, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                violation_logs.append({
                    "box": box,
                    "violation_type": violation_flag
                })

        return frame, violation_logs
