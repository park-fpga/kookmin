import cv2
import numpy as np


class TrafficLightClassifier:

    def _green_mask(self, hsv):
        return cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))

    def _bright_green_mask(self, hsv):
        """켜진 신호등 초록 전용 - 채도/명도 높여서 배경 나무 제거"""
        return cv2.inRange(hsv, (40, 150, 150), (90, 255, 255))

    def _red_mask(self, hsv):
        lo = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
        hi = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        return cv2.bitwise_or(lo, hi)

    def _yellow_mask(self, hsv):
        return cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))

    def _is_arrow_shape(self, mask):
        """초록 마스크의 윤곽선 원형도로 화살표 여부 판별 (원=1.0, 화살표<0.6)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if area < 30 or perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity < 0.6

    def _classify_one(self, img, box):
        """신호등 박스 하나의 상태를 반환"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        bright_green = self._bright_green_mask(hsv)
        if cv2.countNonZero(bright_green) >= 30 and self._is_arrow_shape(bright_green):
            return 'green_arrow'

        counts = {
            'red':    cv2.countNonZero(self._red_mask(hsv)),
            'green':  cv2.countNonZero(self._green_mask(hsv)),
            'yellow': cv2.countNonZero(self._yellow_mask(hsv)),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] >= 30 else 'unknown'

    def classify(self, img, tl_boxes):
        """tl_boxes: {i: box} → {i: state}"""
        return {i: self._classify_one(img, box) for i, box in tl_boxes.items()}

    def draw(self, annotated, tl_boxes, tl_states):
        """annotated 이미지에 신호등 상태 라벨 덧그리기"""
        color_bgr = {
            'red':         (0, 0, 255),
            'green':       (0, 200, 0),
            'green_arrow': (255, 200, 0),
            'yellow':      (0, 200, 255),
            'unknown':     (180, 180, 180),
        }
        for i, state in tl_states.items():
            x1, y1, _, _ = map(int, tl_boxes[i].xyxy[0])
            cv2.putText(annotated, state, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color_bgr.get(state, (255, 255, 255)), 2)
