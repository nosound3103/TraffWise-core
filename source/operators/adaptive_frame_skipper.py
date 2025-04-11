import time


class AdaptiveFrameSkipper:
    def __init__(self, config):
        self.target_fps = config.get("target_fps", 20)

        self.max_skip_rate = config.get("max_skip_rate", 10)
        self.fps_timer = time.time()
        self.frame_time = time.time()

        self.frame_counter = 0
        self.current_fps = 0
        self.total_skip_frames = 0

        self.processing_times = []
        self.history_size = 5

        self.skip_pattern = []
        self.pattern_position = 0

        print(f"Frame skipper initialized with target FPS: {self.target_fps}")

    def is_skipable(self):
        """
        Determine if the current frame should be skipped based on our calculated pattern.
        Returns True if the frame should be skipped, False otherwise.
        """

        if not self.skip_pattern:
            return False

        should_skip = self.skip_pattern[self.pattern_position] == 1

        self.pattern_position = (
            self.pattern_position + 1) % len(self.skip_pattern)

        if should_skip:
            self.total_skip_frames += 1

        return should_skip

    def calculate_skip_pattern(self, video_fps, avg_processing_time):
        """
        Calculate an optimal frame skip pattern based on video FPS and processing time.

        Args:
            video_fps: The source video's frames per second
            avg_processing_time: Average time (seconds) to process one frame
        """
        max_possible_fps = min(1.0 / avg_processing_time, video_fps)

        if max_possible_fps >= self.target_fps:
            self.skip_pattern = [0]
            print(
                f"Processing all frames - can achieve {max_possible_fps:.1f} FPS")
            return

        ratio = video_fps / max_possible_fps

        pattern = []

        total_pattern_frames = min(int(ratio * 10), 60)
        kept_frames = max(int(total_pattern_frames / ratio), 1)

        for i in range(total_pattern_frames):
            if i % (total_pattern_frames / kept_frames) < 1.0:
                pattern.append(0)  # Keep this frame
            else:
                pattern.append(1)  # Skip this frame

        self.skip_pattern = pattern
        self.pattern_position = 0

        # kept = pattern.count(0)
        # total = len(pattern)
        # effective_fps = (video_fps * kept) / total

        # print(f"New skip pattern: keeping {kept}/{total} frames")
        # print(f"Pattern: {pattern}")
        # print(f"Expected effective FPS: {effective_fps:.1f}")

    def adjust_skip_rate(self, processing_time, video_fps):
        """
        Adjust skip pattern based on measured processing time

        Args:
            processing_time: Time in seconds to process the last frame
            video_fps: The source video's frame rate
        """
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.history_size:
            self.processing_times.pop(0)

        avg_time = sum(self.processing_times) / len(self.processing_times)

        if not self.skip_pattern or abs(avg_time - processing_time) > 0.01:
            self.calculate_skip_pattern(video_fps, avg_time)

    def update_fps(self):
        """Update the FPS calculation"""
        self.frame_counter += 1
        elapsed = time.time() - self.fps_timer

        if elapsed >= 1.0:
            self.current_fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.fps_timer = time.time()
