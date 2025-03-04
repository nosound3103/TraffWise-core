import numpy as np

from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, fps: int = 60):
        self.coordinates = defaultdict(lambda: deque(maxlen=30))
        self.fps = fps

    def update_coordinates(self, track_id: int, point: np.ndarray):
        """Store coordinates for speed estimation."""
        self.coordinates[track_id].append(point)

    def calculate_speed(self, track_id: int) -> float:
        """Calculate speed in km/h if there are enough points."""
        if len(self.coordinates[track_id]) < 2:
            return 0.0
        start = self.coordinates[track_id][0]
        end = self.coordinates[track_id][-1]
        distance = np.linalg.norm(np.array(start) - np.array(end))
        time = len(self.coordinates[track_id]) / self.fps
        speed = (distance / time) * 3.6  # Convert m/s to km/h
        return speed
