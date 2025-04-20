import yaml
import os
import cv2


def scale_bboxes(bboxes, new_size, original_size=640):
    """Scale multiple bounding boxes to a new image size.

    Args:
        bboxes (list of tuples): List of (x_min, y_min, x_max, y_max) bounding boxes.
        new_size (tuple): (new_width, new_height) to scale the bboxes to.
        original_size (int): Original image size (default is 640x640).

    Returns:
        list of tuples: Scaled bounding boxes.
    """
    new_w, new_h = new_size
    scale_x = new_w / original_size
    scale_y = new_h / original_size

    return [
        (
            int(x_min * scale_x),
            int(y_min * scale_y),
            int(x_max * scale_x),
            int(y_max * scale_y)
        )
        for x_min, y_min, x_max, y_max in bboxes
    ]


def scale_video(path, new_size):
    """Scale video to a new size.

    Args:
        path (str): Path to the input video.
        new_size (tuple): (new_width, new_height) to scale the video to.

    Returns:
        str: Path to the scaled video.
    """
    output_path = path.replace(".mp4", "_scaled.mp4")
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, new_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        scaled_frame = cv2.resize(frame, new_size)
        out.write(scaled_frame)

    cap.release()
    out.release()

    return output_path


def convert_coordinate(coord, source_size=(1920, 1080), target_size=(1280, 720)):
    """Convert a single coordinate from source to target resolution."""
    x_scale = target_size[0] / source_size[0]
    y_scale = target_size[1] / source_size[1]
    return [int(coord[0] * x_scale), int(coord[1] * y_scale)]


def convert_coordinates(coords_list, source_size=(1920, 1080), target_size=(1280, 720)):
    """Convert a list of coordinates from source to target resolution."""
    if not coords_list:  # Handle empty lists
        return []
    return [convert_coordinate(coord, source_size, target_size) for coord in coords_list]


def process_lane(lane_data):
    """Process all coordinate data in a lane."""
    # Convert polygon coordinates

    if lane_data["polygons"]["coordinates"]:
        lane_data["polygons"]["coordinates"] = convert_coordinates(
            lane_data["polygons"]["coordinates"])

    # Convert expected direction coordinates
    if lane_data["expected_direction"]["coordinates"]:
        lane_data["expected_direction"]["coordinates"] = convert_coordinates(
            lane_data["expected_direction"]["coordinates"]
        )

    # Convert traffic light coordinates if they exist
    if lane_data["traffic_light"]["coordinates"]:
        lane_data["traffic_light"]["coordinates"] = convert_coordinates(
            lane_data["traffic_light"]["coordinates"]
        )

    # Convert stop area coordinates if they exist
    if lane_data["stop_area"]["coordinates"]:
        lane_data["stop_area"]["coordinates"] = convert_coordinates(
            lane_data["stop_area"]["coordinates"]
        )

    return lane_data


class InlineListDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(InlineListDumper, self).increase_indent(flow, indentless)


def resize_coordinates_from_lane(input_file, output_file, source_size=(1920, 1080), target_size=(1280, 720)):
    """Resize coordinates from lane data."""
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    for road_id, road_data in data["roads"].items():
        for lane_id, lane_data in road_data.items():
            data["roads"][road_id][lane_id] = process_lane(lane_data)

    with open(output_file, 'w') as file:
        yaml.dump(data, file, Dumper=InlineListDumper,
                  default_flow_style=True)

# scale_video("data/samples/videos/Traffic_VN.mp4", (1280, 720))
# scale_video("data/samples/videos/wronglane_and_speed.mp4", (1280, 720))
# scale_video("data/samples/videos/wronglane_and_speed_2.mp4", (1280, 720))
# scale_video("data/samples/videos/wronglane_and_speed_3.mp4", (1280, 720))
# scale_video("data/samples/videos/PVB_Vien_Kiem_Sat.mp4", (1280, 720))


# resize_coordinates_from_lane(
#     "data/samples/annotations/Traffic_VN.yml",
#     "data/samples/annotations/Traffic_VN_scaled.yml",
#     source_size=(1920, 1080),
#     target_size=(1280, 720)
# )

# resize_coordinates_from_lane(
#     "data/samples/annotations/wronglane_and_speed.yml",
#     "data/samples/annotations/wronglane_and_speed_scaled.yml",
#     source_size=(1920, 1080),
#     target_size=(1280, 720)
# )

# resize_coordinates_from_lane(
#     "data/samples/annotations/wronglane_and_speed_2.yml",
#     "data/samples/annotations/wronglane_and_speed_2_scaled.yml",
#     source_size=(1920, 1080),
#     target_size=(1280, 720)
# )

# resize_coordinates_from_lane(
#     "data/samples/annotations/wronglane_and_speed_3.yml",
#     "data/samples/annotations/wronglane_and_speed_3_scaled.yml",
#     source_size=(1920, 1080),
#     target_size=(1280, 720)
# )

# resize_coordinates_from_lane(
#     "data/samples/annotations/PVB_Vien_Kiem_Sat.yml",
#     "data/samples/annotations/PVB_Vien_Kiem_Sat_scaled.yml",
#     source_size=(1920, 1080),
#     target_size=(1280, 720)
# )
