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
from recognition.lane import detect_lanes, calculate_steering

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
                
            # 1. 원본 이미지를 인식하기 좋은 사이즈(320x240)로 리사이즈
            frame = cv2.resize(self.image, (320, 240))
            
            # 2. lane.py의 차선 인식 모듈 호출
            result_img, lane_data, debug_imgs = detect_lanes(frame)
            
            # 3. 조향각(Steering) 계산
            # calculate_steering은 -1.0 ~ 1.0 사이의 정규화된 값을 반환합니다.
            steer = calculate_steering(lane_data, 320)
            
            # 4. Xycar 모터 제어 명령 (조향각: -50 ~ 50, 속도: 10 설정)
            angle = float(steer * 50.0)
            speed = float(10)  # 필요에 따라 속도를 가감하세요.
            
            self.drive(angle, speed)
            
            # 5. 차선 인식 결과를 화면에 출력하여 확인 (디버깅 용도)
            cv2.imshow("Lane Tracking", result_img)
            cv2.waitKey(1)
                
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