import time


class AdaptiveFrameSkipper:
    def __init__(self, config):
        self.skip_frames = config["initial_skip"]
        self.target_fps = config["target_fps"]
        self.target_frame_time = 1.0 / self.target_fps
        self.max_skip_frames = config["max_skip_frames"]
        self.adjustment_speed = config["adjustment_speed"]

        self.fps_counter = 0
        self.fps_timer = 0
        self.current_fps = 0
        self.total_skip_frames = 0
        self.frame_counter = 0
        self.processing_times = []

    def update_fps(self):
        self.fps_counter += 1
        if time.time() - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = time.time()

    def adjust_skip_rate(self, processing_time):
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 5:
            self.processing_times.pop(0)

        avg_processing_time = sum(
            self.processing_times) / len(self.processing_times)

        if avg_processing_time > self.target_frame_time:
            target_skip = min(
                int(avg_processing_time / self.target_frame_time),
                self.max_skip_frames
            )
        else:
            target_skip = 1

        self.skip_frames = int(
            self.skip_frames * (1 - self.adjustment_speed) +
            target_skip * self.adjustment_speed
        )

        self.skip_frames = max(1, self.skip_frames)

    def is_skipable(self):
        return self.frame_counter % self.skip_frames != 0
