import cv2
import numpy as np

# ── 원근 변환 포인트 (320×240 기준) ──
SRC_PTS = np.float32([
    [  0, 240],   # 좌하단
    [320, 240],   # 우하단
    [310, 105],   # 우상단
    [ 10, 105],   # 좌상단
])
DST_PTS = np.float32([
    [ 40, 240],
    [280, 240],
    [280,   0],
    [ 40,   0],
])

# ── 슬라이딩 윈도우 파라미터 ──
N_WINDOWS = 9
MARGIN    = 50
MIN_PIX   = 10

_M     = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
_M_INV = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)

_lane_width_cache = [120]


def _lane_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (22, 120, 120), (35, 255, 255))
    white  = cv2.inRange(hsv, (0,   0,  180), (180, 40, 255))
    return yellow, white


def _find_peaks(histogram, n=3, suppress_width=25, min_height=30):
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


def _track_lanes(yellow_warped, white_warped):
    h, w = yellow_warped.shape

    y_hist = np.sum(yellow_warped[h // 2:, :], axis=0).astype(float)
    y_hist[:20] = 0
    y_hist[w - 20:] = 0
    y_peaks = _find_peaks(y_hist, n=1, suppress_width=25, min_height=30)
    left_x = y_peaks[0] if y_peaks else w // 3

    w_hist = np.sum(white_warped[h // 2:, :], axis=0).astype(float)
    w_hist[:20] = 0
    w_hist[w - 20:] = 0
    w_peaks = _find_peaks(w_hist, n=3, suppress_width=25, min_height=30)
    right_candidates = [p for p in w_peaks if p > left_x + 30]
    right_x = max(right_candidates) if right_candidates else min(w - 1, left_x + int(_lane_width_cache[0]))

    win_h = h // N_WINDOWS
    nz_y_y = np.array(yellow_warped.nonzero()[0])
    nz_x_y = np.array(yellow_warped.nonzero()[1])
    nz_y_w = np.array(white_warped.nonzero()[0])
    nz_x_w = np.array(white_warped.nonzero()[1])

    yellow_inds, right_inds = [], []
    lx_cur, rx_cur = left_x, right_x

    debug = np.zeros((h, w, 3), dtype=np.uint8)
    debug[nz_y_y, nz_x_y] = [0, 200, 200]
    debug[nz_y_w, nz_x_w] = [180, 180, 180]

    for win in range(N_WINDOWS):
        y_lo = h - (win + 1) * win_h
        y_hi = h - win * win_h

        ll, lr = int(lx_cur) - MARGIN, int(lx_cur) + MARGIN
        good_y = np.where(
            (nz_y_y >= y_lo) & (nz_y_y < y_hi) &
            (nz_x_y >= ll)   & (nz_x_y < lr)
        )[0]
        yellow_inds.append(good_y)
        cv2.rectangle(debug, (ll, y_lo), (lr, y_hi), (0, 255, 255), 1)

        rl, rr = int(rx_cur) - MARGIN, int(rx_cur) + MARGIN
        good_r = np.where(
            (nz_y_w >= y_lo) & (nz_y_w < y_hi) &
            (nz_x_w >= rl)   & (nz_x_w < rr)
        )[0]
        right_inds.append(good_r)
        cv2.rectangle(debug, (rl, y_lo), (rr, y_hi), (0, 0, 255), 1)

        found_y = len(good_y) > MIN_PIX
        found_r = len(good_r) > MIN_PIX
        if found_y:
            lx_cur = int(np.mean(nz_x_y[good_y]))
        if found_r:
            rx_cur = int(np.mean(nz_x_w[good_r]))

        lane_w = int(_lane_width_cache[0])
        if found_y and not found_r:
            rx_cur = lx_cur + lane_w
        elif found_r and not found_y:
            lx_cur = rx_cur - lane_w

    yellow_inds = np.concatenate(yellow_inds)
    right_inds  = np.concatenate(right_inds)

    yellow_fit = right_fit = None
    y_min_y = y_min_r = h

    if len(yellow_inds) >= 50:
        yellow_fit = np.polyfit(nz_y_y[yellow_inds], nz_x_y[yellow_inds], 2)
        debug[nz_y_y[yellow_inds], nz_x_y[yellow_inds]] = [0, 255, 255]
        y_min_y = int(nz_y_y[yellow_inds].min())
    if len(right_inds) >= 50:
        right_fit = np.polyfit(nz_y_w[right_inds], nz_x_w[right_inds], 2)
        debug[nz_y_w[right_inds], nz_x_w[right_inds]] = [180, 180, 255]
        y_min_r = int(nz_y_w[right_inds].min())

    if yellow_fit is not None and right_fit is not None:
        y_eval = h // 2
        detected_width = float(np.polyval(right_fit, y_eval)) - float(np.polyval(yellow_fit, y_eval))
        if 50 < detected_width < 220:
            _lane_width_cache[0] = detected_width

    y_draw_start = max(y_min_y, y_min_r)
    return yellow_fit, right_fit, debug, y_draw_start


def _draw_lane_overlay(img, ref_warped, left_fit, right_fit, y_draw_start):
    h, w = ref_warped.shape
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
    """차선 인식 순수 퍼셉션 — fit 결과와 디버그 이미지만 반환.
    lane_center 계산은 LaneDrive(control)에서 담당."""
    img = np.ascontiguousarray(image_array)
    h, w = img.shape[:2]

    yellow, white = _lane_masks(img)
    yellow_w = cv2.warpPerspective(yellow, _M, (w, h))
    white_w  = cv2.warpPerspective(white,  _M, (w, h))

    yellow_fit, right_fit, debug_win, y_draw_start = _track_lanes(yellow_w, white_w)

    result = _draw_lane_overlay(img, yellow_w, yellow_fit, right_fit, y_draw_start)

    warped_color = cv2.cvtColor(cv2.bitwise_or(yellow_w, white_w), cv2.COLOR_GRAY2BGR)
    binary_color = cv2.cvtColor(cv2.bitwise_or(yellow, white), cv2.COLOR_GRAY2BGR)

    return result, (yellow_fit, right_fit, w), (warped_color, debug_win, binary_color)
