import yaml
import cv2
import numpy as np
from .lane import Lane
from .intersection import Intersection
from .road import Road
from .lane import Lane

from shapely import Polygon


class RoadManager:
    def __init__(self, config, camera_name):
        self.config = config["samples"][camera_name]
        self.annotation_path = self.config["annotation_path"]
        self.read_annotation()

    def read_annotation(self):
        with open(self.annotation_path, 'r') as f:
            self.data = yaml.safe_load(f)

        self.roads = []
        self.intersection = []
        self.road_config = self.data.get("roads", {})

        for road_key, road_info in self.road_config.items():
            if road_key.startswith("intersection"):
                src_polygon = np.array(
                    road_info["src_polygon"]["coordinates"])
                tgt_polygon = np.array(
                    road_info["tgt_polygon"]["coordinates"])

                intersection_obj = Intersection(
                    src_polygon=src_polygon,
                    tgt_polygon=tgt_polygon,
                    speed_limit=road_info.get("speed_limit", 60)
                )

                self.intersection.append(intersection_obj)

            elif road_key.startswith("road"):
                lanes = []

                src_polygon = np.array(
                    road_info["src_polygon"]["coordinates"])
                tgt_polygon = np.array(
                    road_info["tgt_polygon"]["coordinates"])
                traffic_light = road_info["traffic_light"].get(
                    "coordinates", None)
                stop_area = road_info["stop_area"].get(
                    "coordinates", None)

                road_obj = Road(
                    src_polygon=src_polygon,
                    tgt_polygon=tgt_polygon,
                    traffic_light=traffic_light,
                    stop_area=stop_area
                )

                for lane_key, lane_data in road_info.items():
                    if lane_key.startswith("lane"):
                        coords = lane_data.get("coordinates", None)
                        coords = np.array(coords, dtype=np.float32)

                        direction_coords = lane_data.get(
                            "direction", {})
                        direction_points = (
                            [(tuple(direction_coords[0]),
                              tuple(direction_coords[1]))]
                            if direction_coords and len(direction_coords) >= 2
                            else None
                        )

                        speed_limit = lane_data.get("speed_limit", 60)

                        lane_obj = Lane(
                            coords=coords,
                            direction_points=direction_points,
                            direction=None,
                            speed_limit=speed_limit,
                        )

                        lanes.append(lane_obj)

                road_obj.lanes = lanes
                self.roads.append(road_obj)

    def get_lane(self, position):
        """Find lane based on a position (x, y)."""
        for road in self.roads:
            if road.is_inside(position):
                for lane in road.lanes:
                    if lane.is_inside(position):
                        return lane

        return None

    def get_intersection(self, position):
        """Find intersection based on a position (x, y)."""
        for intersection in self.intersection:
            if intersection.is_inside(position):
                return intersection

        return None

    def get_road(self, position):
        """Find road based on a position (x, y)."""
        for road in self.roads:
            if road.is_inside(position):
                return road

        return None

    def get_speed_limit(self, road_id, lane_id=None):
        """
        Get speed limit for a specific road or lane

        Args:
            road_id (str): ID of the road (e.g., "intersection", "road_1")
            lane_id (str, optional): ID of the lane (e.g., "lane_1"). Defaults to None.

        Returns:
            int: Speed limit value or None if not found
        """
        try:
            # Handle intersection
            if road_id == "intersection":
                return self.road_config["intersection"].get("speed_limit", 60)

            # Handle roads with lanes
            if road_id in self.road_config:
                road_info = self.road_config[road_id]

                # If lane_id is specified, get lane specific speed limit
                if lane_id and lane_id in road_info:
                    return road_info[lane_id].get("speed_limit", 60)

                # If no lane_id, return road's default speed limit
                return road_info.get("speed_limit", 60)

            return None

        except Exception as e:
            print(f"Error getting speed limit for {road_id} - {lane_id}: {e}")
            return None

    def update_speed_limit(self, road_id, speed_limit, lane_id=None):
        """
        Update speed limit for a road or specific lane

        Args:
            road_id (str): ID of the road (e.g., "intersection", "road_1")
            speed_limit (int): New speed limit value
            lane_id (str, optional): ID of the lane (e.g., "lane_1"). Defaults to None.
        """
        try:
            # Handle intersection
            if road_id == "intersection":
                self.road_config["intersection"]["speed_limit"] = speed_limit
                # Also update the intersection object
                if self.intersection:
                    self.intersection[0].speed_limit = speed_limit

            # Handle roads with lanes
            elif road_id in self.road_config:
                road_info = self.road_config[road_id]
                road_idx = int(road_id.split('_')[1]) - 1

                # Update lane specific speed limit
                if lane_id and lane_id in road_info:
                    road_info[lane_id]["speed_limit"] = speed_limit
                    # Also update the lane object
                    if 0 <= road_idx < len(self.roads):
                        lane_idx = int(lane_id.split('_')[1]) - 1
                        if 0 <= lane_idx < len(self.roads[road_idx].lanes):
                            self.roads[road_idx].lanes[lane_idx].speed_limit = speed_limit

                # Update road's default speed limit
                else:
                    road_info["speed_limit"] = speed_limit

        except Exception as e:
            print(f"Error updating speed limit for {road_id} - {lane_id}: {e}")

    def draw_lanes(self, frame):
        """Draw lane annotations"""
        for road in self.roads:
            for lane in road.lanes:
                # Draw lane boundaries
                pts = lane.coordinates.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                if hasattr(lane, 'direction') and lane.direction is not None:
                    mid_point = np.mean(pts, axis=0)[0].astype(np.int32)
                    direction = lane.direction
                    end_point = (
                        int(mid_point[0] + direction[0]),
                        int(mid_point[1] + direction[1])
                    )
                    cv2.arrowedLine(frame, tuple(mid_point),
                                    end_point, (0, 255, 0), 2)

    def draw_roads(self, frame):
        """Draw road boundaries"""
        for road in self.roads:
            # Draw road boundaries
            pts = road.src_polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            # Draw stop areas if they exist
            if road.stop_area is not None:
                stop_pts = np.array(road.stop_area).reshape(
                    (-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [stop_pts], True,
                              (0, 0, 255), 2)

    def draw_intersections(self, frame):
        """Draw intersection areas"""
        for intersection in self.intersection:
            pts = intersection.src_polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

            center = np.mean(pts, axis=0)[0].astype(np.int32)
            cv2.putText(frame, "Intersection",
                        (center[0] - 40, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
