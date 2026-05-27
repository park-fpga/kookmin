import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')

    def detect(self, img):
        """YOLO 검출만 수행. 반환: (annotated_img, tl_boxes)
        tl_boxes: {index: box} - 신호등(class 9) 박스만 모은 딕셔너리"""
        results = self.yolo_model(img, verbose=False, classes=[0, 2, 9], conf=0.65)

        boxes = results[0].boxes
        valid_indices = []
        tl_boxes = {}

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])

            if cls_id == 0 and conf < 0.7:
                continue
            valid_indices.append(i)

            if cls_id == 9:
                tl_boxes[i] = boxes[i]

        annotated = results[0][valid_indices].plot()
        return annotated, tl_boxes
