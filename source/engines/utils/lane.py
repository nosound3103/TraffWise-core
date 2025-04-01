import numpy as np
from .view_transformer import ViewTransformer


class Lane:
    def __init__(self, source_polygon: np.ndarray, target_width: int, target_height: int,
                 expected_direction_points: np.ndarray, expected_direction: np.ndarray,
                 traffic_light: np.ndarray, stop_area: np.ndarray, speed_limit: int):
        self.source_polygon = source_polygon.astype(np.float32)
        self.target_width = target_width
        self.target_height = target_height
        self.target_polygon = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ], dtype=np.float32)
        self.transformer = ViewTransformer(
            self.source_polygon, self.target_polygon)
        self.traffic_light = traffic_light
        self.stop_area = stop_area
        self.speed_limit = speed_limit
        self.traffic_light_status = None

        if expected_direction_points is not None:
            pA, pB = expected_direction_points[0]
            self.expected_direction = self.compute_expected_direction(pA, pB)
        elif expected_direction is not None:
            v = np.array(expected_direction, dtype=np.float32)
            norm = np.linalg.norm(v)
            self.expected_direction = v / \
                norm if norm > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)
        else:
            raise ValueError(
                "Provide expected_direction_points or expected_direction.")

    def compute_expected_direction(self, pointA, pointB) -> np.ndarray:
        v = np.array(pointB, dtype=np.float32) - \
            np.array(pointA, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)

    def transform(self, points: np.ndarray) -> np.ndarray:
        return self.transformer.transform_points(points)
