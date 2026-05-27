import numpy as np


ANGLE_GAIN = 2   # P 게인: 차선 오차(픽셀) → 조향각
D_GAIN     = 1   # D 게인: 오차 변화율 → 방향 전환 예측 보정
ANGLE_MAX  = 200.0
ALPHA      = 0.9  # 스무딩 (낮출수록 빠른 반응)


class LaneDrive:

    def __init__(self):
        self.last_angle       = 0.0
        self.last_lane_center = None
        self.last_error       = 0.0

    def update(self, left_fit, right_fit, lane_center: float, img_width: int) -> float:
        lane_detected = (left_fit is not None) or (right_fit is not None)

        if lane_detected:
            self.last_lane_center = lane_center
        elif self.last_lane_center is not None:
            lane_center = self.last_lane_center

        error      = lane_center - img_width / 2
        error_diff = error - self.last_error   # 오차 변화율 (S자 구간에서 방향 전환 예측)
        self.last_error = error

        raw_angle = ANGLE_GAIN * error + D_GAIN * error_diff
        smooth    = ALPHA * raw_angle + (1 - ALPHA) * self.last_angle
        angle     = float(np.clip(smooth, -ANGLE_MAX, ANGLE_MAX))

        self.last_angle = angle
        return angle
