from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSORT:
    def __init__(self,
                 config,

                 max_age=15):

        self.config = config
        self.conf_threshold = config["conf_threshold"]
        self.tracker = DeepSort(
            max_age=max_age, max_iou_distance=0.90, n_init=2)

        self.class_names = list(config["labels"].keys())

    def convert_box(self, box):
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]

    def extract_detections(self, boxes):
        detections = [
            [self.convert_box(box),
             conf,
             label]
            for [label, conf, box] in boxes if conf > self.conf_threshold]

        return detections

    def update_tracks(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
