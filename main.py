# main.py
import os, warnings, time
import cv2
from client_lib import GetStatus, GetRaw, AVControl, CloseSocket

from modules import config as C
from modules.vision import load_models, get_lane_mask, get_traffic_sign
from modules.control import PIDController, calculate_steering_angle, calculate_speed
from modules.state_machine import update as fsm_update, reset_all

# Optional: disable console logs for cleaner output
if C.SIMPLE_LOG:
    import builtins as _bi
    def print(*args, **kwargs):
        return None


def main():
    """Main control loop: model loading, inference, FSM update, and vehicle control."""
    load_models()
    pid = PIDController(C.PID_KP, C.PID_KI, C.PID_KD)
    reset_all()

    # Create display windows if enabled
    if C.SHOW_WINDOWS:
        cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
        cv2.namedWindow("segment", cv2.WINDOW_NORMAL)

    last_loop_time = time.time()

    try:
        while True:
            # --- Time step ---
            now = time.time()
            dt = now - last_loop_time
            last_loop_time = now

            # --- Frame & telemetry ---
            state = GetStatus()
            raw = GetRaw()
            if raw is None:
                AVControl(speed=0, angle=0)
                if C.SHOW_WINDOWS and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break
                continue

            try:
                speed_now_kmh = float(state.get("Speed", 0.0))
            except Exception:
                speed_now_kmh = 0.0
            distance_this_frame = (speed_now_kmh * 1000.0 / 3600.0) * dt

            # --- Object detection (YOLO) ---
            raw_vis = raw.copy() if (C.DRAW_OVERLAY or C.SHOW_WINDOWS) else raw
            raw_vis, sign_text, box_info, yolo_updated = get_traffic_sign(raw_vis)
            if C.SHOW_WINDOWS:
                cv2.imshow("raw", raw_vis)

            # --- Lane segmentation ---
            seg = get_lane_mask(raw)
            if seg is None:
                AVControl(speed=0, angle=0)
                if C.SHOW_WINDOWS and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break
                continue

            # --- Steering control (PID) ---
            # NOTE: this also draws three diagnostic dots on 'seg' when DRAW_OVERLAY=True.
            angle_pid, _err = calculate_steering_angle(seg, pid)

            # Display segmentation after overlay has been drawn
            if C.SHOW_WINDOWS:
                cv2.imshow("segment", seg)

            # --- FSM decision logic ---
            angle_sm, speed_override, _mode, _pre = fsm_update(
                sign_text, box_info, angle_pid, distance_this_frame, pid
            )

            # --- Speed control ---
            # Keep speed as float for smooth logging; send integer to the controller.
            if speed_override is not None:
                speed_cmd_f = float(speed_override)
            else:
                speed_cmd_f = float(calculate_speed(seg, angle_for_ema=angle_pid))

            AVControl(speed=int(round(speed_cmd_f)), angle=float(angle_sm))

            # --- Logging ---
            sign_log = sign_text if sign_text else "N/A"
            print(f"[{_mode}] speed={speed_cmd_f:4.1f}  angle={angle_sm:6.2f}  sign={sign_log}")

            if C.SHOW_WINDOWS and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        CloseSocket()
        if C.SHOW_WINDOWS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

