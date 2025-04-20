import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, source, target, method=cv2.RANSAC, ransac_thresh=5.0):
        """
        source: Nx2 array of source points
        target: Nx2 array of target points
        method: method for findHomography (e.g., cv2.RANSAC, cv2.LMEDS)
        ransac_thresh: RANSAC reprojection threshold
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Use findHomography instead of getPerspectiveTransform
        self.matrix, self.mask = cv2.findHomography(
            source, target, method, ransac_thresh)

    def transform_points(self, points):
        if points.size == 0 or self.matrix is None:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.matrix)

        return transformed_points.reshape(-1, 2)
