#!/usr/bin/env python3
# -*- coding: utf-8 -*- 1
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import rclpy, time, cv2, os, math
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge
from recognition.yolo import YoloDetector
from recognition.lane import detect_lanes
from judgement.traffic_light_classifier import TrafficLightClassifier
from control.traffic_light_drive import TrafficLightDrive
from control.lane_drive import LaneDrive

#=============================================
# ROS2 Node 클래스 정의
#=============================================
class TrackDriverNode(Node):

    #=============================================
    # 클래스 생성 초기화 함수
    #=============================================
    def __init__(self):

        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')
        
        # 상수값 및 초기값 설정
        self.image = None  # 카메라 토픽 데이터를 저장할 변수
        self.motor_msg = XycarMotor()  # 모터토픽 메시지
        self.lidar_ranges = None
        self.bridge = CvBridge()
        self.yolo = YoloDetector()
        self.classifier = TrafficLightClassifier()
        self.traffic = TrafficLightDrive()
        self.lane_ctrl = LaneDrive()
        self.yolo_tick = 0          # YOLO 실행 주기 카운터
        self.cached_speed = 10.0   # 마지막으로 계산된 속도 캐시
        self.smooth_factor = 1.0   # 커브 속도 감쇠 계수 (스무딩 적용)
        
        # ROS2 Publisher & Subscriber 설정
        self.motor_pub = self.create_publisher(XycarMotor,'xycar_motor',10)
        
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
		
        self.get_logger().info("Track Driver Node Initialized")
              
    #=============================================
    # 카메라 토픽을 수신하는 콜백 함수
    #=============================================
    def cam_callback(self, data):
        # 수신한 메시지를 OpenCV 이미지로 변환하여 저장
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    #=============================================
    # 라이다 토픽을 수신하는 콜백 함수
    #=============================================
    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges   
      
    #=============================================
    # 모터제어 토픽을 발행하는 Publisher 함수
    #=============================================
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    #=============================================
    # 메인 루프
    #=============================================
    def main_loop(self):
    
        self.get_logger().info("======================================")
        self.get_logger().info("  S T A R T    D R I V I N G ...      ")
        self.get_logger().info("======================================")

        while rclpy.ok():
            # ROS2 콜백 함수들이 실행될 수 있도록 이벤트를 한 번씩 처리합니다.
            rclpy.spin_once(self, timeout_sec=0.01)
        
            # 아직 카메라 이미지가 수신되지 않았다면 대기합니다.
            if self.image is None:
                continue

            frame = cv2.resize(self.image, (320, 240))

            # 차선 인식 → 조향각 계산
            lane_result, (left_fit, right_fit, lane_center, w, current_lane), (warped_color, debug_win, binary_color) = detect_lanes(frame)
            angle = self.lane_ctrl.update(left_fit, right_fit, lane_center, w)
            cv2.imshow("Lane Detection", lane_result)
            cv2.imshow("Sliding Window", debug_win)
            cv2.imshow("Bird Eye View", warped_color)
            cv2.waitKey(1)

            
            self.yolo_tick += 1
            if self.yolo_tick % 5 == 0:
                annotated, tl_boxes = self.yolo.detect(frame)
                tl_states = self.classifier.classify(frame, tl_boxes)
                self.classifier.draw(annotated, tl_boxes, tl_states)
                self.traffic.update(tl_states)
                self.cached_speed = self.traffic.get_speed()
                self.traffic.show_debug(annotated)

            # 급격한 커브에서 속도 감소 — 속도 회복은 천천히 (급가속 방지)
            base_speed = self.cached_speed
            if base_speed > 0:
                target_factor = max(0.4, 1.0 - abs(angle) / 100.0)
                # 감속은 빠르게(0.3), 가속 회복은 느리게(0.05) → 커브 후 급가속 방지
                if target_factor < self.smooth_factor:
                    self.smooth_factor = 0.3 * target_factor + 0.7 * self.smooth_factor
                else:
                    self.smooth_factor = 0.05 * target_factor + 0.95 * self.smooth_factor
                speed = base_speed * self.smooth_factor
            else:
                speed = 0.0
                self.smooth_factor = 1.0

            self.drive(angle=angle, speed=speed)


                
#=============================================
# 메인 함수
#=============================================
def main(args=None):
      
    rclpy.init(args=args)
    node = TrackDriverNode()
	
    try:
        # main_loop() 함수를 호출하여 실행합니다.
        node.main_loop()
    except KeyboardInterrupt:
        # 사용자 인터럽트 (Ctrl+C)가 발생하면 예외를 처리합니다.
        pass
    finally:
        # 노드를 종료하고 ROS2를 정리합니다.
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()