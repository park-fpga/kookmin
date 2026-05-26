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
    dst = np.float32([
        [w * 0.2, 0],
        [w * 0.8, 0],
        [w * 0.8, h],
        [w * 0.2, h]
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
    h, w = binary.shape

    histogram = np.sum(binary[h // 2:, :], axis=0)
    midpoint = w // 2

    # 이미지 가장자리 12%는 인도/경계 오인식 방지용으로 탐색 제외
    edge_margin = int(w * 0.12)
    left_hist  = histogram[edge_margin:midpoint]
    right_hist = histogram[midpoint:w - edge_margin]

    left_peak  = int(np.max(left_hist))  if len(left_hist)  else 0
    right_peak = int(np.max(right_hist)) if len(right_hist) else 0
    left_x  = int(np.argmax(left_hist))  + edge_margin if left_peak  > 50 else midpoint // 2
    right_x = int(np.argmax(right_hist)) + midpoint    if right_peak > 50 else midpoint + midpoint // 2

    n_windows = 9
    window_height = h // n_windows
    margin = 80
    min_pix = 50

    # 슬라이딩 윈도우가 인도 쪽으로 흘러가지 않도록 x 경계 설정
    LEFT_X_MIN, LEFT_X_MAX   = edge_margin,       int(w * 0.55)
    RIGHT_X_MIN, RIGHT_X_MAX = int(w * 0.45), w - edge_margin

    nonzero = binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_current = left_x
    right_current = right_x
    left_lane_idx = []
    right_lane_idx = []

    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for window in range(n_windows):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height
        x_left_low = left_current - margin
        x_left_high = left_current + margin
        x_right_low = right_current - margin
        x_right_high = right_current + margin

        cv2.rectangle(debug_img, (x_left_low, y_low), (x_left_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(debug_img, (x_right_low, y_low), (x_right_high, y_high), (0, 255, 0), 2)

        good_left = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                     (nonzero_x >= x_left_low) & (nonzero_x < x_left_high)).nonzero()[0]
        good_right = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                      (nonzero_x >= x_right_low) & (nonzero_x < x_right_high)).nonzero()[0]

        left_lane_idx.append(good_left)
        right_lane_idx.append(good_right)

        if len(good_left) > min_pix:
            left_current = int(np.clip(np.mean(nonzero_x[good_left]), LEFT_X_MIN, LEFT_X_MAX))
        if len(good_right) > min_pix:
            right_current = int(np.clip(np.mean(nonzero_x[good_right]), RIGHT_X_MIN, RIGHT_X_MAX))

    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    left_x_pts = nonzero_x[left_lane_idx]
    left_y_pts = nonzero_y[left_lane_idx]
    right_x_pts = nonzero_x[right_lane_idx]
    right_y_pts = nonzero_y[right_lane_idx]

    left_fit = None
    right_fit = None

    if len(left_x_pts) > 100:
        left_fit = np.polyfit(left_y_pts, left_x_pts, 2)
        last_left_fit[0] = left_fit
        stale_left[0] = 0
    else:
        stale_left[0] += 1
        if stale_left[0] > MAX_STALE:
            last_left_fit[0] = None
        elif last_left_fit[0] is not None:
            left_fit = last_left_fit[0]

    if len(right_x_pts) > 100:
        right_fit = np.polyfit(right_y_pts, right_x_pts, 2)
        last_right_fit[0] = right_fit
        stale_right[0] = 0
    else:
        stale_right[0] += 1
        if stale_right[0] > MAX_STALE:
            last_right_fit[0] = None
        elif last_right_fit[0] is not None:
            right_fit = last_right_fit[0]

    # 두 차선이 모두 검출됐을 때 폭 검증:
    # 폭이 너무 크면(인도까지 포함한 것) 픽셀 수가 적은 쪽(= 덜 확실한 쪽) 버리기
    if left_fit is not None and right_fit is not None:
        y_check = h * 0.8
        lx = left_fit[0]  * y_check**2 + left_fit[1]  * y_check + left_fit[2]
        rx = right_fit[0] * y_check**2 + right_fit[1] * y_check + right_fit[2]
        if rx - lx > w * 0.65:
            if len(left_x_pts) >= len(right_x_pts):
                right_fit = None
                last_right_fit[0] = None
            else:
                left_fit = None
                last_left_fit[0] = None

    return left_fit, right_fit, debug_img

def draw_lanes(img, left_fit, right_fit, Minv):
    h, w = img.shape[:2]
    result = img.copy()

    if left_fit is None and right_fit is None:
        return result, None

    ploty = np.linspace(0, h - 1, h)
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
    left_fit, right_fit, debug_img = sliding_window(binary, img)
    result, lane_center = draw_lanes(img, left_fit, right_fit, Minv)

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
    Kp = 0.0035  # 비례 게인 (현재 오차 반영)
    Kd = 0.0050  # 미분 게인 (오차 변화량 반영하여 흔들림 억제)
    
    derivative = error - last_error[0]
    last_error[0] = error
    
    raw_steer = (Kp * error) + (Kd * derivative)
    raw_steer = max(-1.0, min(1.0, raw_steer))

    # EMA 스무딩: 이전 steer 60% + 현재 40% → 급격한 조향 방지
    smoothed = 0.6 * last_steer[0] + 0.4 * raw_steer
    last_steer[0] = smoothed
    return smoothed
