import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')

    def _classify_traffic_light(self, img, box):
        """신호등 박스 영역에서 빨강/초록/노랑 색상을 분석해 상태를 반환"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 빨강: 0~10 또는 160~180 (HSV에서 빨강은 두 구간에 걸침)
        red_lo = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
        red_hi = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_lo, red_hi)

        # 초록: 40~90
        green_mask = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))

        # 노랑: 15~35
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))

        counts = {
            'red':    cv2.countNonZero(red_mask),
            'green':  cv2.countNonZero(green_mask),
            'yellow': cv2.countNonZero(yellow_mask),
        }

        best = max(counts, key=counts.get)
        # 픽셀이 너무 적으면 불확실
        if counts[best] < 30:
            return 'unknown'
        return best

    def detect(self, img):
        results = self.yolo_model(img, verbose=False, classes=[0, 2, 9], conf=0.5)

        boxes = results[0].boxes
        valid_indices = []
        traffic_light_states = {}  # index -> 'red'|'green'|'yellow'|'unknown'

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf  = float(boxes.conf[i])

            if cls_id == 0 and conf < 0.7:
                continue
            valid_indices.append(i)

            if cls_id == 9:  # traffic light
                state = self._classify_traffic_light(img, boxes[i])
                traffic_light_states[i] = state

        annotated = results[0][valid_indices].plot()

        # 신호등 박스 위에 색상 라벨 덧그리기
        color_bgr = {'red': (0, 0, 255), 'green': (0, 200, 0),
                     'yellow': (0, 200, 255), 'unknown': (180, 180, 180)}
        for i, state in traffic_light_states.items():
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
            cv2.putText(annotated, state, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color_bgr.get(state, (255, 255, 255)), 2)

        return annotated, traffic_light_states