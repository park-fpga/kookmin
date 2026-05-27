import cv2


class TrafficLightDrive:

    def __init__(self):
        self.state = 'unknown'

    def update(self, tl_states: dict):
        if not tl_states:
            return
        states = list(tl_states.values())
        self.state = 'red' if 'red' in states else states[0]

    def get_speed(self, default_speed: float = 10.0) -> float:
        if self.state in ('red', 'yellow'):
            return 0.0
        return default_speed

    def show_debug(self, annotated_img):
        color = (0, 0, 255) if self.state == 'red' else (0, 200, 0)
        cv2.putText(annotated_img, f"Traffic: {self.state}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("YOLO Detection", annotated_img)
        cv2.waitKey(1)
