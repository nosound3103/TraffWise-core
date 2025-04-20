import numpy as np
import cv2
from .view_transformer import ViewTransformer


class Road:
    def __init__(
            self,
            src_polygon: np.ndarray,
            tgt_polygon: np.ndarray,
            traffic_light: np.ndarray,
            stop_area: np.ndarray):

        self.src_polygon = src_polygon.astype(np.float32)
        self.tgt_polygon = tgt_polygon
        self.transformer = ViewTransformer(
            self.src_polygon, self.tgt_polygon)
        self.traffic_light = traffic_light
        self.stop_area = stop_area
        self.traffic_light_status = None
        self.lanes = []

    def transform(self, points: np.ndarray) -> np.ndarray:
        return self.transformer.transform_points(points)

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the road polygon.
        """
        return cv2.pointPolygonTest(self.src_polygon, tuple(point), False) >= 0
