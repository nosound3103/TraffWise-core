import yaml
import cv2
import numpy as np
from .lane import Lane
from shapely import Polygon


class LaneManager:
    def __init__(self, config, camera_name):
        self.config = config
        self.annotation_path = self.config["samples"][camera_name]["annotation_path"]
        self.lanes = self.create_lane_list()
        self.intersect = self.compute_intersections()

    def create_lane_list(self):
        with open(self.annotation_path, 'r') as f:
            data = yaml.safe_load(f)

        lanes = []
        roads = data.get("roads", {})

        for road_key, road_info in roads.items():
            for lane_key, lane_data in road_info.items():
                poly_coords = lane_data.get(
                    "polygons", {}).get("coordinates", None)
                if poly_coords is None or len(poly_coords) < 4:
                    continue

                source_polygon = np.array(poly_coords, dtype=np.float32)

                ed_coords = lane_data.get(
                    "expected_direction", {}).get("coordinates", None)
                expected_direction_points = (
                    [(tuple(ed_coords[0]), tuple(ed_coords[1]))]
                    if ed_coords and len(ed_coords) >= 2
                    else None
                )

                target_info = lane_data.get("TARGET", {})
                target_width = target_info.get("width", 0)
                target_height = target_info.get("height", 0)

                speed_limit = lane_data.get("speed_limit", 60)

                traffic_coords = lane_data.get(
                    "traffic_light", {}).get("coordinates", None)
                if traffic_coords is None or len(traffic_coords) < 4:
                    traffic_light = None
                else:
                    traffic_light = np.array(traffic_coords, dtype=np.float32)

                stop_coords = lane_data.get(
                    "stop_area", {}).get("coordinates", None)
                if stop_coords is None or len(stop_coords) < 4:
                    stop_area = None
                else:
                    stop_area = np.array(stop_coords, dtype=np.float32)

                lane_obj = Lane(
                    source_polygon=source_polygon,
                    target_width=int(target_width),
                    target_height=int(target_height),
                    expected_direction_points=expected_direction_points,
                    expected_direction=None,
                    traffic_light=traffic_light,
                    stop_area=stop_area,
                    speed_limit=int(speed_limit)
                )
                lanes.append(lane_obj)
        return lanes

    def get_lane(self, position):
        """Find lane based on a position (x, y)."""
        for lane in self.lanes:
            if cv2.pointPolygonTest(lane.source_polygon, position, False) >= 0:
                return lane
        return None

    def compute_intersections(self):
        """
        Computes the intersection of all lane polygons using Shapely.
        If the intersection is a MultiPolygon, it splits it into individual Polygon objects.
        """
        if not self.lanes:
            intersect = []
            return intersect

        shapely_polys = [Polygon(lane.source_polygon) for lane in self.lanes]

        inter_poly = shapely_polys[0]
        for poly in shapely_polys[1:]:
            inter_poly = inter_poly.intersection(poly)
            if inter_poly.is_empty:
                intersect = []
                return intersect

        if inter_poly.geom_type == "MultiPolygon":
            intersect = list(inter_poly.geoms)
        else:
            intersect = [inter_poly]
        return intersect

    def is_point_in_intersection(self, position):
        """
        Checks whether the given position (x, y) lies in any of the computed intersection regions.
        Uses cv2.pointPolygonTest on the approximated contour (exterior coordinates) of each intersection.
        """
        if not self.intersect:
            self.compute_intersections()
        for poly in self.intersect:
            contour = np.array(poly.exterior.coords, dtype=np.int32)
            if cv2.pointPolygonTest(contour, position, False) >= 0:
                return True
        return False

    def draw_lanes(self, frame):
        for lane in self.lanes:
            pts = lane.source_polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
