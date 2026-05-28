import numpy as np

# ── BEV 기준 차선 중앙 계산 파라미터 ──
YELLOW_TARGET   = 100   # 2차선: 노란선 목표 x (왼쪽)
YELLOW_TARGET_1 = 205   # 1차선: 노란선 목표 x (오른쪽)
WHITE_OFFSET_2  = 70    # 2차선: 오른쪽 흰선 기준 차선 중앙까지 거리
WHITE_OFFSET_1  = 70    # 1차선: 오른쪽 흰선 기준 보정 거리

Y_EVAL = 160            # BEV에서 다항식 평가 y 위치 (320×240 기준 h//1.5)


class LaneJudgement:

    def __init__(self):
        self._current_lane = 2       # 현재 주행 차선 (1 or 2)
        self._initialized  = False   # 최초 차선 자동 감지 완료 여부
        self._lane_width   = 120.0   # 차선 폭 캐시 (픽셀)

    def set_lane(self, lane: int):
        """외부 명령(차선 변경, 회피 등)으로 차선을 변경할 때 호출"""
        self._current_lane = int(lane)
        self._initialized  = True

    @property
    def current_lane(self) -> int:
        return self._current_lane

    def update(self, yellow_fit, right_fit, img_width: int) -> float:
        """fit 결과로부터 현재 차선 기준 BEV 차선 중앙 x 좌표 반환"""
        w = img_width

        if yellow_fit is not None:
            yellow_x = float(np.polyval(yellow_fit, Y_EVAL))

            # 최초 1회 차선 자동 감지
            if not self._initialized:
                self._current_lane = 2 if yellow_x < w / 2 else 1
                self._initialized  = True

            if right_fit is not None:
                right_x = float(np.polyval(right_fit, Y_EVAL))
                detected_width = right_x - yellow_x
                if 50 < detected_width < 220:
                    self._lane_width = detected_width

            if self._current_lane == 2:
                return yellow_x + (w / 2 - YELLOW_TARGET)
            else:
                return yellow_x - (YELLOW_TARGET_1 - w / 2)

        elif right_fit is not None:
            right_x = float(np.polyval(right_fit, Y_EVAL))
            if self._current_lane == 2:
                return right_x - WHITE_OFFSET_2
            else:
                return right_x - WHITE_OFFSET_1 - self._lane_width

        return w / 2
