# vision.py
import os, warnings, cv2, numpy as np
from typing import Optional, Tuple, Dict, Any

from . import config as C

_ai_ready = False
DEVICE = "cpu"
_use_fp16 = False
_buf = None
_model = None

_yolo = None
YOLO_NAMES: Dict[int, str] = {}
YOLO_FRAME_STRIDE = C.YOLO_FRAME_STRIDE_DEFAULT
_last_yolo_result = None
_yolo_frame_counter = 0


def load_models() -> None:
    global _ai_ready, DEVICE, _use_fp16, _buf, _model, _yolo, YOLO_NAMES, YOLO_FRAME_STRIDE
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message=r".*weights_only.*")

    try:
        import torch, segmentation_models_pytorch as smp
        torch.backends.cudnn.benchmark = True
        use_cuda = torch.cuda.is_available()
        DEVICE = "cuda" if use_cuda else "cpu"

        _model = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=1)
        _model.load_state_dict(torch.load(C.MODEL_PATH, map_location="cpu"))
        _model = _model.to(DEVICE).eval()

        if use_cuda:
            try:
                props = torch.cuda.get_device_properties(0)
                if getattr(props, "major", 0) >= 7:
                    _use_fp16 = True
                    _model = _model.half()
                _buf = torch.empty((1, 1, C.IMG_H, C.IMG_W), device=DEVICE,
                                   dtype=(torch.float16 if _use_fp16 else torch.float32))
                try:
                    _model = torch.compile(_model)
                except Exception:
                    pass
            except Exception:
                _reset_cpu_torch()
        else:
            _buf = _empty_buf_cpu()

        try:
            torch.set_num_threads(2)
        except Exception:
            pass
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        _ai_ready = True
    except Exception:
        _reset_cpu_torch()

    # YOLO
    try:
        from ultralytics import YOLO
        _yolo = YOLO(C.YOLO_PATH)
        YOLO_NAMES.update(_yolo.names)
        if DEVICE == "cuda":
            import torch
            props = torch.cuda.get_device_properties(0)
            weak_gpu = (props.multi_processor_count <= 16)
            if weak_gpu:
                YOLO_FRAME_STRIDE = max(3, YOLO_FRAME_STRIDE)
    except Exception:
        _yolo = None


def _reset_cpu_torch():
    global _ai_ready, DEVICE, _use_fp16, _buf, _model
    DEVICE = "cpu"; _use_fp16 = False; _ai_ready = False
    try:
        import torch
        _model = None
        _buf = _empty_buf_cpu()
    except Exception:
        _buf = None


def _empty_buf_cpu():
    import torch
    return torch.empty((1, 1, C.IMG_H, C.IMG_W), device="cpu", dtype=torch.float32)


# --- Public API ---

def get_lane_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return lane mask as BGR (same size as input), or None on failure."""
    if not _ai_ready or _model is None:
        return None
    try:
        import torch
        h_raw, w_raw = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_rz = cv2.resize(gray, (C.IMG_W, C.IMG_H), interpolation=cv2.INTER_LINEAR)
        np_in = (gray_rz.astype(np.float32) / 255.0 - 0.5) / 0.5
        if _use_fp16 and DEVICE == "cuda":
            _buf[0, 0].copy_(torch.from_numpy(np_in).to(DEVICE).half())
        else:
            _buf[0, 0].copy_(torch.from_numpy(np_in).to(DEVICE))
        with torch.inference_mode():
            logits = _model(_buf)
            mask_u8 = (logits > C.AI_THRESH_LOGIT).squeeze().to(torch.uint8).mul_(255).contiguous()
            mask = mask_u8.cpu().numpy()
            seg_gray = cv2.resize(mask, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)
            return cv2.cvtColor(seg_gray, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


def get_traffic_sign(bgr: np.ndarray) -> Tuple[np.ndarray, Optional[str], Optional[Dict[str, float]], bool]:
    """Run YOLO. Returns: (visual_bgr, sign_text, box_info{area,conf}, yolo_updated)"""
    global _last_yolo_result, _yolo_frame_counter
    if _yolo is None:
        return bgr, None, None, False
    _yolo_frame_counter += 1
    yolo_updated = False
    if _yolo_frame_counter % YOLO_FRAME_STRIDE == 0:
        try:
            import torch
            res = _yolo.predict(bgr, imgsz=C.YOLO_IMGSZ, conf=C.YOLO_CONF, iou=C.YOLO_IOU,
                                half=(_use_fp16 and DEVICE == "cuda"), verbose=False,
                                device=(0 if DEVICE == "cuda" else "cpu"))[0]
            _last_yolo_result = res
            yolo_updated = True
        except Exception:
            res = _last_yolo_result
    else:
        res = _last_yolo_result
    if res is None:
        return bgr, None, None, yolo_updated

    sign_text, box_info = None, None
    if len(res.boxes):
        try:
            i = int(res.boxes.conf.argmax().item())
            xyxy = res.boxes.xyxy[i].detach().cpu().numpy().astype(int)
            cls_id = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            label = YOLO_NAMES.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = xyxy.tolist()
            sign_text = label
            box_info = {"area": (x2 - x1) * (y2 - y1), "conf": conf}
            if C.DRAW_OVERLAY:
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(bgr, f"{label} {conf:.2f}", (x1, max(15,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        except Exception:
            pass
    return bgr, sign_text, box_info, yolo_updated
