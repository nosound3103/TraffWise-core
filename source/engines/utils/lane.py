import numpy as np
import cv2


class Lane:
    def __init__(
            self,
            coords: np.ndarray,
            direction_points: np.ndarray,
            direction: np.ndarray,
            speed_limit: int):

        self.coordinates = coords.astype(np.float32)
        self.speed_limit = speed_limit
        self.traffic_light_status = None

        if direction_points is not None:
            pA, pB = direction_points[0]
            self.direction = self.compute_direction(pA, pB)
        elif direction is not None:
            v = np.array(direction, dtype=np.float32)
            norm = np.linalg.norm(v)
            self.direction = v / \
                norm if norm > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)
        else:
            raise ValueError(
                "Provide direction_points or expected_direction.")

    def compute_direction(self, pointA, pointB) -> np.ndarray:
        v = np.array(pointB, dtype=np.float32) - \
            np.array(pointA, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else np.array([0.0, 0.0])

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the lane polygon.
        """
        return cv2.pointPolygonTest(self.coordinates, tuple(point), False) >= 0
