import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, source, target):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.matrix)

        return transformed_points.reshape(-1, 2)
