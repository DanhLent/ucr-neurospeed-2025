# state_machine.py
import time, numpy as np
from typing import Optional, Tuple, Dict

from . import config as C
from .control import PIDController

# Internal state
_last_turn_sign: Optional[str] = None
_distance_since_sign_disappeared = 0.0
_distance_in_turn = 0.0
_is_turning = False
_pending_sign: Optional[str] = None

# Go-straight guards
_gs_seen_cnt = 0
_gs_last_area = 0
_gs_seen_yolo_cnt = 0
_gs_pre_guard_on = False
_gs_guard_on = False
_gs_guard_dist = 0.0
_gs_guard_target = 0.0
_gs_kind = "X"  # X/T placeholder
_gs_last_seen_t = 0.0

# Current chosen turn delay
_turn_delay_current = C.TURN_DELAY_METERS_TURN_RIGHT


def reset_all():
    global _last_turn_sign, _distance_since_sign_disappeared, _distance_in_turn, _is_turning
    global _pending_sign, _gs_seen_cnt, _gs_last_area, _gs_seen_yolo_cnt, _gs_pre_guard_on
    global _gs_guard_on, _gs_guard_dist, _gs_guard_target, _gs_kind, _gs_last_seen_t
    global _turn_delay_current
    _last_turn_sign = None
    _distance_since_sign_disappeared = 0.0
    _distance_in_turn = 0.0
    _is_turning = False
    _pending_sign = None
    _gs_seen_cnt = 0
    _gs_last_area = 0
    _gs_seen_yolo_cnt = 0
    _gs_pre_guard_on = False
    _gs_guard_on = False
    _gs_guard_dist = 0.0
    _gs_guard_target = 0.0
    _gs_kind = "X"
    _gs_last_seen_t = 0.0
    _turn_delay_current = C.TURN_DELAY_METERS_TURN_RIGHT


class Mode:
    OFF = "OFF"
    PRE = "PRE"
    VIS = "VIS"
    WAIT_TURN = "WAIT-TURN"
    TURN = "TURN"
    POST = "POST"


def update(sign_text: Optional[str], box_info: Optional[Dict[str, float]],
           angle_pid: float, distance_this_frame: float,
           pid: PIDController) -> Tuple[float, Optional[int], str, bool]:
    """
    Returns (angle_override, speed_override, mode, preturn_guard_active)
    - angle_override: may differ from angle_pid if we clamp/force
    - speed_override: None means "use base speed"; otherwise an int speed
    - mode: for HUD/logging
    - preturn_guard_active: whether PRE-TURN guard engaged
    """
    global _last_turn_sign, _distance_since_sign_disappeared, _distance_in_turn, _is_turning
    global _pending_sign, _gs_seen_cnt, _gs_last_area, _gs_seen_yolo_cnt, _gs_pre_guard_on
    global _gs_guard_on, _gs_guard_dist, _gs_guard_target, _gs_kind, _gs_last_seen_t
    global _turn_delay_current

    now = time.time()
    angle = angle_pid
    speed_override: Optional[int] = None
    preturn_guard_active = False

    # Hard states first
    if _is_turning:
        angle = (-C.TURN_ANGLE if _last_turn_sign == "turn_left" else C.TURN_ANGLE)
        _distance_in_turn += distance_this_frame
        if _distance_in_turn >= C.TURN_DURATION_METERS:
            _is_turning = False
            _last_turn_sign = None
            _turn_delay_current = C.TURN_DELAY_METERS_TURN_RIGHT
            pid.reset()
        return angle, 15, Mode.TURN, False

    if _last_turn_sign is not None:
        _distance_since_sign_disappeared += distance_this_frame
        if _distance_since_sign_disappeared >= _turn_delay_current:
            _is_turning = True
            _distance_in_turn = 0.0
            angle = (-C.TURN_ANGLE if _last_turn_sign == "turn_left" else C.TURN_ANGLE)
            return angle, 15, Mode.TURN, False
        else:
            angle = (-C.WAIT_PREBIAS_ANGLE if _last_turn_sign == "turn_left" else C.WAIT_PREBIAS_ANGLE)
            return angle, 20, Mode.WAIT_TURN, False

    # No hard state: inspect sign
    is_any_sign_visible = (sign_text is not None and box_info is not None)
    area = box_info.get("area", 0) if box_info else 0

    turn_actionable = is_any_sign_visible and (sign_text != "go_straight") and (area > C.SIGN_AREA_THRESHOLD)
    go_actionable   = is_any_sign_visible and (sign_text == "go_straight")

    if turn_actionable:
        if _pending_sign is None:
            _pending_sign = sign_text
        if area >= C.SIGN_PREP_AREA:
            preturn_guard_active = True
            want_left  = (sign_text == "turn_left") or (sign_text == "no_right")
            want_right = (sign_text == "turn_right") or (sign_text == "no_left")
            if want_left:
                angle = float(np.clip(angle - C.PRETURN_PREBIAS_ANGLE, -C.PRETURN_MAX_ANGLE, C.PRETURN_MAX_ANGLE))
            elif want_right:
                angle = float(np.clip(angle + C.PRETURN_PREBIAS_ANGLE, -C.PRETURN_MAX_ANGLE, C.PRETURN_MAX_ANGLE))
            else:
                angle = float(np.clip(angle, -C.PRETURN_MAX_ANGLE, C.PRETURN_MAX_ANGLE))
            speed_override = int(C.PRETURN_SPEED_CAP)
            return angle, speed_override, Mode.PRE, True
    else:
        if (not is_any_sign_visible) and _pending_sign is not None:
            if _pending_sign == "no_left":
                _last_turn_sign = "turn_right"; _turn_delay_current = C.TURN_DELAY_METERS_NO_LEFT
            elif _pending_sign == "no_right":
                _last_turn_sign = "turn_left";  _turn_delay_current = C.TURN_DELAY_METERS_NO_RIGHT
            elif _pending_sign == "turn_left":
                _last_turn_sign = "turn_left";  _turn_delay_current = C.TURN_DELAY_METERS_TURN_LEFT
            elif _pending_sign == "turn_right":
                _last_turn_sign = "turn_right"; _turn_delay_current = C.TURN_DELAY_METERS_TURN_RIGHT
            _distance_since_sign_disappeared = 0.0
            _pending_sign = None

        if go_actionable:
            _gs_last_seen_t = now
            _gs_seen_cnt += 1
            _gs_last_area = area
            _gs_seen_yolo_cnt += 1  # caller should increment only when YOLO updated; simplified here
            if (not _gs_pre_guard_on) and (_gs_seen_yolo_cnt >= C.GSTRAIGHT_SEEN_YOLO_MIN) and (_last_turn_sign is None):
                _gs_pre_guard_on = True
            angle_limit = C.GSTRAIGHT_PRE_MAX_ANGLE if _gs_pre_guard_on else C.GSTRAIGHT_VISIBLE_MAX_ANGLE
            angle = float(np.clip(angle_pid, -angle_limit, angle_limit))
            speed_override = int(C.STRAIGHT_SPEED_CAP)
            return angle, speed_override, (Mode.PRE if _gs_pre_guard_on else Mode.VIS), False
        else:
            just_missed = ((time.time() - _gs_last_seen_t) <= C.GSTRAIGHT_MISS_TIMEOUT)
            can_post_guard = (_gs_pre_guard_on or (_gs_seen_yolo_cnt >= C.GSTRAIGHT_SEEN_YOLO_MIN)
                              or (_gs_seen_cnt >= C.GSTRAIGHT_SEEN_MIN and _gs_last_area >= C.GSTRAIGHT_AREA_ON))
            if (not _gs_guard_on) and just_missed and can_post_guard and (_last_turn_sign is None):
                _gs_kind = "X"
                _gs_guard_target = (C.GSTRAIGHT_METERS_T if _gs_kind == "T" else C.GSTRAIGHT_METERS_X)
                _gs_guard_on = True; _gs_guard_dist = 0.0
            _gs_seen_yolo_cnt = 0
            _gs_seen_cnt = 0; _gs_last_area = 0; _gs_pre_guard_on = False
            if _gs_guard_on:
                angle = float(np.clip(angle, -C.GSTRAIGHT_MAX_ANGLE, C.GSTRAIGHT_MAX_ANGLE))
                _gs_guard_dist += distance_this_frame
                if _gs_guard_dist >= _gs_guard_target:
                    _gs_guard_on = False
                return angle, None, f"POST-{_gs_kind}", False

    # Slight clamp when we are waiting for turn decision
    if _pending_sign == "turn_left":
        angle = min(angle,  C.OPPOSITE_LIMIT)
    elif _pending_sign == "turn_right":
        angle = max(angle, -C.OPPOSITE_LIMIT)

    return angle, None, Mode.OFF, preturn_guard_active
