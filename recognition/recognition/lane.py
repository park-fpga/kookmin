import cv2
import numpy as np

# ── 원근 변환 포인트 (320×240 기준, 시뮬레이션 카메라 각도에 맞게 조정) ──
# SRC: 원본 이미지의 사다리꼴 (도로 영역)
# DST: 변환 후 직사각형 (버드아이뷰)
SRC_PTS = np.float32([
    [  0, 240],   # 좌하단
    [320, 240],   # 우하단
    [260, 115],   # 우상단 (커브 대응을 위해 넓게)
    [ 60, 115],   # 좌상단
])
DST_PTS = np.float32([
    [ 40, 240],
    [280, 240],
    [280,   0],
    [ 40,   0],
])

# ── 스트립 탐색 파라미터 ──
N_STRIPS = 5 # BEV 수평 분할 수 (줄일수록 스트립이 두꺼워져 점선 간격 영향 감소)
MARGIN   = 70   # 각 스트립 탐색 좌우 폭 (넓을수록 급커브 추적 가능)

_M     = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
_M_INV = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)

# 현재 주행 차선의 폭 캐시 (한 쪽만 보일 때 사용)
_lane_width_cache = [120]
# 현재 인식된 차선 번호 (1 또는 2)
_current_lane = [2]


def _binary_mask(img):
    """흰색/노란색 차선 픽셀 추출 → 이진 마스크"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, (0,   0,  180), (180,  40, 255))
    yellow = cv2.inRange(hsv, (15, 100,  100), ( 35, 255, 255))
    return cv2.bitwise_or(white, yellow)


def _find_peaks(histogram, n=3, suppress_width=25, min_height=30):
    """히스토그램에서 상위 n개 피크를 x 오름차순으로 반환"""
    hist = histogram.copy().astype(float)
    peaks = []
    for _ in range(n):
        idx = int(np.argmax(hist))
        if hist[idx] < min_height:
            break
        peaks.append(idx)
        lo = max(0, idx - suppress_width)
        hi = min(len(hist), idx + suppress_width)
        hist[lo:hi] = 0
    return sorted(peaks)


def _strip_centroids(binary_warped):
    """BEV를 수평 스트립으로 분할해 히스토그램 가중 평균으로 차선 중심 추출 후 2차 다항식 피팅"""
    h, w = binary_warped.shape

    # 전체 히스토그램으로 초기 차선 위치 + 차선 번호 결정
    histogram = np.sum(binary_warped[h // 2:, :], axis=0).astype(float)
    histogram[:20] = 0
    histogram[w - 20:] = 0
    peaks = _find_peaks(histogram, n=3, suppress_width=25, min_height=30)

    car_bev_x = w // 2
    if len(peaks) >= 3:
        p0, p1, p2 = peaks[0], peaks[1], peaks[2]
        if car_bev_x <= p1:
            _current_lane[0] = 1
            left_x, right_x = p0, p1
        else:
            _current_lane[0] = 2
            left_x, right_x = p1, p2
    elif len(peaks) == 2:
        left_x, right_x = peaks[0], peaks[1]
    elif len(peaks) == 1:
        p = peaks[0]
        half = _lane_width_cache[0]
        if p < w // 2:
            left_x, right_x = p, min(w - 1, p + half)
        else:
            left_x, right_x = max(0, p - half), p
    else:
        left_x, right_x = w // 4, w * 3 // 4

    strip_h = h // N_STRIPS
    lx_cur, rx_cur = left_x, right_x
    left_pts, right_pts = [], []

    debug = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

    for i in range(N_STRIPS):
        y_lo  = h - (i + 1) * strip_h
        y_hi  = h - i * strip_h
        y_mid = (y_lo + y_hi) // 2

        strip_hist = np.sum(binary_warped[y_lo:y_hi, :], axis=0).astype(float)

        # 좌측: lx_cur 주변 MARGIN 범위에서 히스토그램 가중 평균
        ll = max(0, int(lx_cur) - MARGIN)
        lr = min(w, int(lx_cur) + MARGIN)
        region_l = strip_hist[ll:lr]
        cv2.rectangle(debug, (ll, y_lo), (lr, y_hi), (0, 255, 0), 1)
        if region_l.sum() > 0:
            lx_cur = int(np.average(np.arange(ll, lr), weights=region_l))
            left_pts.append((y_mid, lx_cur))
            cv2.circle(debug, (lx_cur, y_mid), 4, (0, 255, 0), -1)

        # 우측: rx_cur 주변 MARGIN 범위에서 히스토그램 가중 평균
        rl = max(0, int(rx_cur) - MARGIN)
        rr = min(w, int(rx_cur) + MARGIN)
        region_r = strip_hist[rl:rr]
        cv2.rectangle(debug, (rl, y_lo), (rr, y_hi), (0, 0, 255), 1)
        if region_r.sum() > 0:
            rx_cur = int(np.average(np.arange(rl, rr), weights=region_r))
            right_pts.append((y_mid, rx_cur))
            cv2.circle(debug, (rx_cur, y_mid), 4, (0, 0, 255), -1)

    left_fit = right_fit = None
    y_min_l = y_min_r = h

    if len(left_pts) >= 3:
        ys, xs = zip(*left_pts)
        left_fit = np.polyfit(ys, xs, 2)
        y_min_l = min(ys)

    if len(right_pts) >= 3:
        ys, xs = zip(*right_pts)
        right_fit = np.polyfit(ys, xs, 2)
        y_min_r = min(ys)

    y_draw_start = max(y_min_l, y_min_r)
    return left_fit, right_fit, debug, y_draw_start


def _draw_lane_overlay(img, binary_warped, left_fit, right_fit, y_draw_start):
    """실제 검출된 y 범위(y_draw_start~h)에서만 차선 영역을 역변환해서 표시"""
    h, w = binary_warped.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    if left_fit is not None and right_fit is not None:
        plot_y = np.linspace(y_draw_start, h - 1, h - y_draw_start)
        lx = np.polyval(left_fit,  plot_y)
        rx = np.polyval(right_fit, plot_y)
        pts_left  = np.array([np.stack([lx, plot_y], axis=1)], dtype=np.int32)
        pts_right = np.array([np.stack([rx, plot_y], axis=1)[::-1]], dtype=np.int32)
        pts = np.hstack([pts_left, pts_right])
        cv2.fillPoly(overlay, pts, (0, 200, 0))

    unwarped = cv2.warpPerspective(overlay, _M_INV, (w, h))
    return cv2.addWeighted(img, 1.0, unwarped, 0.4, 0)


def detect_lanes(image_array):
    img = np.ascontiguousarray(image_array)
    h, w = img.shape[:2]

    binary = _binary_mask(img)
    warped = cv2.warpPerspective(binary, _M, (w, h))

    left_fit, right_fit, debug_win, y_draw_start = _strip_centroids(warped)

    # 앞쪽(h//3)에서 평가 — 양쪽/한쪽 모두 커브가 반영된 위치를 봐야 방향이 맞음
    y_eval = h // 1.5
    if left_fit is not None and right_fit is not None:
        lx = np.polyval(left_fit,  y_eval)
        rx = np.polyval(right_fit, y_eval)
        lane_center = (lx + rx) / 2
        _lane_width_cache[0] = rx - lx
    elif left_fit is not None:
        half = _lane_width_cache[0] / 2
        lane_center = np.polyval(left_fit, y_eval) + half
    elif right_fit is not None:
        half = _lane_width_cache[0] / 2
        lane_center = np.polyval(right_fit, y_eval) - half
    else:
        lane_center = w / 2

    result = _draw_lane_overlay(img, warped, left_fit, right_fit, y_draw_start)

    # BEV (lane_center, y_eval) → 원본 이미지 좌표로 역변환하여 표시
    bev_pt = np.array([[[float(lane_center), float(y_eval)]]], dtype=np.float32)
    orig_pt = cv2.perspectiveTransform(bev_pt, _M_INV)
    vis_x = int(np.clip(orig_pt[0, 0, 0], 0, w - 1))
    vis_y = int(np.clip(orig_pt[0, 0, 1], 0, h - 1))
    cv2.circle(result, (vis_x, vis_y), 7, (0, 255, 255), -1)
    cv2.circle(result, (w // 2, h - 10), 5, (255, 0,   0), -1)

    # 디버그 텍스트
    det = f"Lane{_current_lane[0]} L:{'O' if left_fit is not None else 'X'} R:{'O' if right_fit is not None else 'X'}  ctr:{int(lane_center)}"
    cv2.putText(result, det, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return result, (left_fit, right_fit, lane_center, w), (warped_color, debug_win, binary_color)
