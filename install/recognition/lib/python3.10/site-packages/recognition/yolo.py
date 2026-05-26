from ultralytics import YOLO

class YoloDetector:
    def __init__(self):
        # YOLOv8 모델 초기화
        self.yolo_model = YOLO('yolov8n.pt')

    def detect(self, img):
        # 전체 기본 신뢰도는 0.5로 설정하여 예측합니다.
        results = self.yolo_model(img, verbose=False, classes=[0, 2, 9], conf=0.5)
        
        boxes = results[0].boxes
        valid_indices = []
        
        # 예측된 박스들을 하나씩 검사하여 조건에 맞는 것만 남깁니다.
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            
            # 사람은 신뢰도가 0.7(70%) 미만이면 무시합니다 (라바콘 오인식 방지)
            if cls_id == 0 and conf < 0.7:
                continue
            valid_indices.append(i)
            
        # 필터링된 결과로만 화면에 그리도록 Results 객체를 업데이트 후 plot() 호출
        return results[0][valid_indices].plot()