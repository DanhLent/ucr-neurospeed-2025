# control.py
import math, time, cv2, numpy as np
from typing import Optional, Tuple
from . import config as C


class PIDController:
    """Simple PID controller for steering angle regulation."""
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = time.time()

    def reset(self) -> None:
        """Reset controller state (integral and previous error/time)."""
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = time.time()

    def update(self, error: float) -> float:
        """Compute PID output for a given lateral error."""
        t = time.time()
        dt = t - self._last_time
        if dt <= 0:
            return 0.0
        p = self.Kp * error
        self._integral += self.Ki * error * dt
        d = self.Kd * (error - self._prev_error) / max(1e-6, dt)
        out = p + self._integral + d
        self._prev_error = error
        self._last_time = t
        return out


# -------- Helpers (lane sampling & curvature) --------

def _center_x_at_row(gray: np.ndarray, y: int, thr: int = 200) -> Optional[int]:
    """Return center x at a horizontal scanline if two lane borders exist."""
    y = int(np.clip(y, 0, gray.shape[0] - 1))
    xs = np.where(gray[y, :] >= thr)[0]
    return int((xs[0] + xs[-1]) // 2) if xs.size >= 2 else None


def _sample_centerline(gray: np.ndarray):
    """Sample lane centers from bottom to top with fixed step."""
    pts, y, taken = [], gray.shape[0] - 1, 0
    while y >= 10 and taken < C.N_ROWS:
        cx = _center_x_at_row(gray, y)
        if cx is not None:
            pts.append((y, cx))
            taken += 1
        y -= C.ROW_STEP
    return pts


def _fit_quadratic_x_of_y(points):
    """Fit x(y) = ay^2 + by + c; return coefficients or None."""
    if len(points) < C.MIN_POINTS_FIT:
        return None
    ys = np.array([p[0] for p in points], dtype=np.float32)
    xs = np.array([p[1] for p in points], dtype=np.float32)
    try:
        return np.polyfit(ys, xs, 2)
    except Exception:
        return None


def _curvature_at_y(coeffs, y: float):
    """Return curvature kappa and radius R at row y."""
    a, b, _ = coeffs
    xp = 2.0 * a * y + b
    xpp = 2.0 * a
    denom = (1.0 + xp * xp) ** 1.5
    if denom <= 1e-6:
        return 0.0, float("inf")
    kappa = abs(xpp) / denom
    return (0.0, float("inf")) if kappa < 1e-9 else (kappa, 1.0 / kappa)


# -------- Steering (PID on lateral error) --------

def calculate_steering_angle(seg_bgr: np.ndarray, pid: PIDController) -> Tuple[float, float]:
    """
    Compute steering angle using PID on lateral error.
    Also draws three diagnostic dots on 'seg_bgr' if DRAW_OVERLAY=True:
      - near lane center (yellow) at CHECKPOINT
      - far lane center (orange) at CHECKPOINT_FAR
      - image center (blue) at CHECKPOINT
    """
    gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)

    def _center_at_band(y_center: int) -> int:
        rows = (y_center - 10, y_center, y_center + 10)
        centers, H = [], gray.shape[0]
        for y in rows:
            y0 = int(np.clip(y, 0, H - 1))
            xs = np.where(gray[y0, :] == 255)[0]
            if xs.size >= 2:
                centers.append(int((xs[0] + xs[-1]) // 2))
        return int(np.mean(centers)) if centers else seg_bgr.shape[1] // 2

    lane_center_near = _center_at_band(C.CHECKPOINT)
    lane_center_far = _center_at_band(C.CHECKPOINT_FAR)
    lane_center = int(0.9 * lane_center_near + 0.1 * lane_center_far)

    error = lane_center - (seg_bgr.shape[1] // 2)
    angle_pid = float(np.clip(pid.update(error), -25.0, 25.0))

    if C.DRAW_OVERLAY:
        H, W = seg_bgr.shape[:2]
        cx_mid = W // 2
        # Only three dots (no lines/arrows)
        cv2.circle(seg_bgr, (lane_center_near, C.CHECKPOINT), 5, (0, 255, 255), -1)
        cv2.circle(seg_bgr, (lane_center_far, C.CHECKPOINT_FAR), 5, (0, 165, 255), -1)
        cv2.circle(seg_bgr, (cx_mid, C.CHECKPOINT), 5, (255, 0, 0), -1)

    return angle_pid, float(error)


# -------- Speed (choose_speed) --------
_rinv_ema, _ang_ema = None, None
_curve_cnt, _straight_cnt = 0, 0
_prev_speed_cmd, _straight_on = None, False


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def _speed_from_R(R):
    if not math.isfinite(R):
        return C.STRAIGHT_SPEED_CAP
    if R <= 90:
        return C.CURVE_SPEED_FLOOR
    if R <= 120:
        return _lerp(38.0, 39.0, (R - 90) / 30.0)
    if R <= 150:
        return _lerp(39.0, 40.5, (R - 120) / 30.0)
    if R <= 200:
        return _lerp(40.5, 41.5, (R - 150) / 50.0)
    if R <= 260:
        return _lerp(41.5, 44.0, (R - 200) / 60.0)
    return C.STRAIGHT_SPEED_CAP


def calculate_speed(seg_bgr: np.ndarray, angle_for_ema: Optional[float] = None) -> float:
    """
    Compute target speed based on lane curvature with simple hysteresis.
    Returns a float for smoother logging; caller may round to int for actuation.
    """
    global _prev_speed_cmd, _straight_on, _rinv_ema, _ang_ema, _curve_cnt, _straight_cnt

    gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
    if int(np.max(gray)) <= 0:
        return float(C.V_MIN)

    pts = _sample_centerline(gray)
    coeffs = _fit_quadratic_x_of_y(pts)
    if coeffs is None:
        return float(C.V_MIN)

    _, R = _curvature_at_y(coeffs, float(C.CHECKPOINT))
    R_inv = 0.0 if (not math.isfinite(R) or R <= 0.0) else 1.0 / R

    if _rinv_ema is None:
        _rinv_ema = R_inv
        _ang_ema = abs(float(angle_for_ema or 0.0))
    else:
        _rinv_ema = (1 - C.EMA_ALPHA_RINV) * _rinv_ema + C.EMA_ALPHA_RINV * R_inv
        _ang_ema = (1 - C.EMA_ALPHA_ANG) * _ang_ema + C.EMA_ALPHA_ANG * abs(float(angle_for_ema or 0.0))

    entering_curve = (_rinv_ema >= C.STRAIGHT_R_EXIT_INV) or (_ang_ema >= C.ENTER_ANG)
    exiting_curve = (_rinv_ema <= C.STRAIGHT_R_ENTER_INV) and (_ang_ema <= C.EXIT_ANG)

    if entering_curve:
        _curve_cnt += 1
        _straight_cnt = 0
    elif exiting_curve:
        _straight_cnt += 1
        _curve_cnt = 0
    else:
        _curve_cnt = max(0, _curve_cnt - 1)
        _straight_cnt = max(0, _straight_cnt - 1)

    if _curve_cnt >= C.CONFIRM_N:
        _straight_on = False
    elif _straight_cnt >= C.CONFIRM_N:
        _straight_on = True

    if _straight_on:
        target = C.STRAIGHT_SPEED_CAP
        v_a, v_rl = C.V_SMOOTH_A_STRAIGHT, C.V_RATE_LIMIT_STRAIGHT
    else:
        target = max(_speed_from_R(R), C.CURVE_SPEED_FLOOR)
        v_a, v_rl = C.V_SMOOTH_A_CURVE, C.V_RATE_LIMIT_CURVE

    target = float(max(C.V_MIN, min(C.V_MAX, target)))

    if _prev_speed_cmd is None:
        _prev_speed_cmd = target

    # First-order smoothing + rate limiting
    target = v_a * target + (1 - v_a) * _prev_speed_cmd
    delta = np.clip(target - _prev_speed_cmd, -v_rl, v_rl)
    _prev_speed_cmd += delta

    return float(_prev_speed_cmd)  # return float (no rounding)

