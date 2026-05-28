import numpy as np

ANGLE_GAIN = 4
D_GAIN     = 1
ANGLE_MAX  = 250.0
ALPHA      = 0.7


class LaneDrive:

    def __init__(self):
        self.last_angle       = 0.0
        self.last_lane_center = None
        self.last_error       = 0.0

    def update(self, lane_center: float, img_width: int, lane_detected: bool) -> float:
        if lane_detected:
            self.last_lane_center = lane_center
        elif self.last_lane_center is not None:
            lane_center = self.last_lane_center

        error      = lane_center - img_width / 2
        error_diff = error - self.last_error
        self.last_error = error

        raw_angle = ANGLE_GAIN * error + D_GAIN * error_diff
        smooth    = ALPHA * raw_angle + (1 - ALPHA) * self.last_angle
        angle     = float(np.clip(smooth, -ANGLE_MAX, ANGLE_MAX))

        self.last_angle = angle
        return angle
