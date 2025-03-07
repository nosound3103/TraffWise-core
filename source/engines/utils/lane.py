
import cv2
import numpy as np

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform image points to target points in real world.."""
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

class Lane:
    def __init__(self, source_polygon: np.ndarray, target_width: int, target_height: int, expected_direction_points=None, expected_direction=None, speed_limit: int=60):
        """
        :param source_polygon: Numpy array (4,2) contains coordinates of image.
        :param target_width: Width of target.
        :param target_height: Height of target.
        :param expected_direction: Expected direction vertor of lane.
        """
        self.source_polygon = source_polygon.astype(np.float32)
        self.target_width = target_width
        self.target_height = target_height
        self.target_polygon = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ], dtype=np.float32)
        self.transformer = ViewTransformer(self.source_polygon, self.target_polygon)
        self.speed_limit = speed_limit

        if expected_direction_points is not None:
            pA, pB = expected_direction_points[0]
            self.expected_direction = self.compute_expected_direction(pA, pB)
        elif expected_direction is not None:
            v = np.array(expected_direction, dtype=np.float32)
            norm = np.linalg.norm(v)
            self.expected_direction = v / norm if norm > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)
        else:
            raise ValueError("Provide expected_direction_points or expected_direction.")

    def compute_expected_direction(self, pointA, pointB):
        v = np.array(pointB, dtype=np.float32) - np.array(pointA, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)

    def transform(self, points: np.ndarray) -> np.ndarray:
        return self.transformer.transform_points(points)


