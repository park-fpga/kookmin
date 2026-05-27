import cv2
import numpy as np

last_left_fit = [None]
last_right_fit = [None]
last_lane_width = [280]   # 버드아이뷰 기준 차선 폭 추정값 (EMA로 갱신)
last_steer = [0.0]
last_error = [0.0]        # PD 제어용 이전 에러값
stale_left = [0]
stale_right = [0]
MAX_STALE = 25            # N 프레임 연속 미검출 시 fit 초기화

def bird_eye_view(img):
    h, w = img.shape[:2]
    src = np.float32([
        [w * 0.35, h * 0.57],
        [w * 0.65, h * 0.57],
        [w * 0.92, h * 0.95],
        [w * 0.08, h * 0.95]
    ])
    # 옆 차선이 보이면서도 해상도를 유지하도록 중앙 50% 영역(0.25~0.75)으로 설정
    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.75, h],
        [w * 0.25, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, Minv

def get_binary(img, v_min, s_max, clahe_limit):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(clahe_limit, 0.1), tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, v_min]), np.array([180, s_max, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))

    binary = cv2.bitwise_or(white_mask, yellow_mask)
    return binary

def sliding_window(binary, img):
    global last_left_fit, last_right_fit, stale_left, stale_right
    h, w = binary.shape

    histogram = np.sum(binary[h // 2:, :], axis=0)

    # scipy.signal.find_peaks를 대체하는 Numpy 기반 피크 파인딩
    height_threshold = 50
    distance = w // 6
    local_maxima = []

    # 1. 주변보다 높은 피크(극댓값) 찾기
    for i in range(1, len(histogram) - 1):
        if histogram[i] >= height_threshold and histogram[i] > histogram[i - 1] and histogram[i] >= histogram[i + 1]:
            local_maxima.append((i, histogram[i]))
            
    # 2. 높이 순으로 정렬하여 높은 피크부터 거리(distance) 제약 확인 후 추가
    local_maxima.sort(key=lambda x: x[1], reverse=True)
    peaks = []
    for idx, val in local_maxima:
        if all(abs(idx - p) >= distance for p in peaks):
            peaks.append(idx)
    peaks.sort() # x축 좌표 순으로 다시 정렬

    n_windows = 9
    window_height = h // n_windows
    margin = 50
    min_pix = 50

    nonzero = binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    detected_fits = []

    for peak in peaks:
        current_x = peak
        lane_idx = []
        
        for window in range(n_windows):
            y_low = h - (window + 1) * window_height
            y_high = h - window * window_height
            x_low = current_x - margin
            x_high = current_x + margin
            
            cv2.rectangle(debug_img, (x_low, y_low), (x_high, y_high), (0, 255, 0), 2)
            
            good_idx = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                        (nonzero_x >= x_low) & (nonzero_x < x_high)).nonzero()[0]
            lane_idx.append(good_idx)
            
            if len(good_idx) > min_pix:
                current_x = int(np.mean(nonzero_x[good_idx]))
                
        lane_idx = np.concatenate(lane_idx)
        if len(lane_idx) > 100:
            fit = np.polyfit(nonzero_y[lane_idx], nonzero_x[lane_idx], 2)
            detected_fits.append(fit)

    # 하단(x 절편)을 기준으로 왼쪽부터 오른쪽으로 차선을 정렬
    detected_fits.sort(key=lambda fit: fit[0] * h**2 + fit[1] * h + fit[2])
    
    # 타겟 차선 선택 로직
    left_fit, right_fit = None, None
    
    # TODO: 회피 제어 플래그 등에 따라 사용할 인덱스를 변경할 수 있습니다.
    # 현재는 기본 주행을 위해 차량의 중심(w/2)을 감싸는 두 선을 잡도록 되어있습니다.
    for fit in detected_fits:
        x_pos = fit[0] * h**2 + fit[1] * h + fit[2]
        if x_pos < w // 2:
            left_fit = fit  # 중앙보다 왼쪽에 있는 가장 가까운 선
        elif x_pos >= w // 2 and right_fit is None:
            right_fit = fit # 중앙보다 오른쪽에 있는 첫 번째 선

    # 3. EMA(지수 이동 평균) 필터 및 이전 프레임 기억(Stale) 로직 복원 (차선 춤추는 현상 방지)
    alpha = 0.7  # 급격한 커브에서 늦게 따라오는 현상을 없애기 위해 현재 프레임 반영 비율을 70%로 높임

    if left_fit is not None:
        if last_left_fit[0] is not None:
            left_fit = alpha * left_fit + (1.0 - alpha) * last_left_fit[0]
        last_left_fit[0] = left_fit
        stale_left[0] = 0
    else:
        stale_left[0] += 1
        if stale_left[0] <= MAX_STALE:
            left_fit = last_left_fit[0]
        else:
            last_left_fit[0] = None

    if right_fit is not None:
        if last_right_fit[0] is not None:
            right_fit = alpha * right_fit + (1.0 - alpha) * last_right_fit[0]
        last_right_fit[0] = right_fit
        stale_right[0] = 0
    else:
        stale_right[0] += 1
        if stale_right[0] <= MAX_STALE:
            right_fit = last_right_fit[0]
        else:
            last_right_fit[0] = None

    # 시각화를 위해 찾은 모든 차선(detected_fits)도 함께 반환
    return left_fit, right_fit, debug_img, detected_fits

def draw_lanes(img, left_fit, right_fit, Minv, all_fits=None):
    h, w = img.shape[:2]
    result = img.copy()

    ploty = np.linspace(0, h - 1, h)

    # 탐지된 모든 차선(옆 차선 포함)을 원본 이미지에 빨간색 선으로 시각화 (디버깅/확인용)
    if all_fits is not None:
        for fit in all_fits:
            fit_x = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
            # 좌표를 묶고 변환 행렬(Minv)을 통해 버드아이뷰에서 원래 시점으로 투영
            pts_line = np.array([np.vstack([fit_x, ploty]).T], dtype=np.float32)
            pts_line_unwarped = cv2.perspectiveTransform(pts_line, Minv)
            cv2.polylines(result, np.int32(pts_line_unwarped), isClosed=False, color=(0, 0, 255), thickness=4)

    if left_fit is None and right_fit is None:
        return result, None

    overlay = np.zeros_like(img)

    # 맨 아랫줄(차 바로 앞) 대신 60% 지점을 기준으로 조향 계산 (더 안정적)
    y_eval = int(h * 0.6)

    if left_fit is not None and right_fit is not None:
        left_x = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_x = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(overlay, np.int32(pts), (0, 255, 0))
        unwarped = cv2.warpPerspective(overlay, Minv, (w, h))
        result = cv2.addWeighted(result, 1, unwarped, 0.3, 0)

        left_eval = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        right_eval = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]

        # 차선 폭 EMA 업데이트 (이상한 값은 무시)
        width = right_eval - left_eval
        if 100 < width < 600:
            last_lane_width[0] = int(0.85 * last_lane_width[0] + 0.15 * width)

        lane_center = (left_eval + right_eval) / 2

    elif left_fit is not None:
        left_eval = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        lane_center = left_eval + last_lane_width[0] / 2
    else:
        right_eval = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
        lane_center = right_eval - last_lane_width[0] / 2

    return result, lane_center

def detect_lanes(image_array):
    img = np.ascontiguousarray(image_array)
    h, w = img.shape[:2]

    v_min = 200
    s_max = 25
    clahe_limit = 1.0

    warped, Minv = bird_eye_view(img)
    binary = get_binary(warped, v_min, s_max, clahe_limit)
    left_fit, right_fit, debug_img, all_fits = sliding_window(binary, img)
    result, lane_center = draw_lanes(img, left_fit, right_fit, Minv, all_fits)

    if left_fit is not None and right_fit is not None:
        status = 'Lane Detected'
        color = (0, 255, 0)
    elif left_fit is not None or right_fit is not None:
        status = 'Partial Lane'
        color = (0, 255, 255)
    else:
        status = 'No Lane'
        color = (0, 0, 255)

    cv2.putText(result, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(result, f'V_min:{v_min} S_max:{s_max} CLAHE:{clahe_limit:.1f} W:{last_lane_width[0]}',
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return result, (left_fit, right_fit, lane_center, w), (warped, binary, debug_img)

def calculate_steering(lane_data, width):
    if lane_data is None:
        return 0.0

    left_fit, right_fit, lane_center, w = lane_data

    if lane_center is None:
        return last_steer[0] * 0.95  # 차선을 순간적으로 놓쳤을 때 핸들을 서서히 풂

    image_center = w / 2
    error = lane_center - image_center
    
    # PD 제어 (Proportional-Derivative) 적용
    Kp = 0.020  # 해상도 축소(320x240)로 오차값이 작아졌으므로 게인을 대폭 증가
    Kd = 0.025  # 미분 게인도 함께 증가
    
    derivative = error - last_error[0]
    last_error[0] = error
    
    raw_steer = (Kp * error) + (Kd * derivative)
    raw_steer = max(-1.0, min(1.0, raw_steer))

    # EMA 스무딩: 커브에서 더 빠르게 반응하도록 현재 값의 비중을 70%로 높임
    smoothed = 0.3 * last_steer[0] + 0.7 * raw_steer
    last_steer[0] = smoothed
    return smoothed
