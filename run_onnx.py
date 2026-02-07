#!/usr/bin/env python3
"""Run exported stitch ONNX model on video frames (GPU by default)."""
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from stitch_seg import compute_static_roi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ONNX stitch inferencer on a video.")
    p.add_argument("--onnx-step", type=str, default="try.onnx", help="Path to step/update ONNX.")
    p.add_argument("--onnx-init", type=str, default="try_init.onnx", help="Path to first_frame ONNX. Defaults to <onnx-step stem>_init.onnx")
    p.add_argument("--video", type=str, default="video01.mp4", help="Input video path.")
    p.add_argument("--start-frame", type=int, default=28000)
    p.add_argument("--end-frame", type=int, default=29000)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--out-video", type=str, default="onnx_overlay.mp4")
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument("--debug", action="store_true", help="Render and save debug video.")
    p.add_argument("--fp16", action="store_true", help="Use FP16 for float16 model inputs/states.")
    return p.parse_args()


def _shape_to_list(shape):
    vals = []
    for d in shape:
        vals.append(d if isinstance(d, int) else None)
    return vals


def _np_dtype_from_ort_type(ort_type: str):
    if ort_type == "tensor(float16)":
        return np.float16
    if ort_type == "tensor(float)":
        return np.float32
    if ort_type == "tensor(double)":
        return np.float64
    if ort_type == "tensor(int64)":
        return np.int64
    if ort_type == "tensor(int32)":
        return np.int32
    if ort_type == "tensor(int8)":
        return np.int8
    if ort_type == "tensor(uint8)":
        return np.uint8
    if ort_type == "tensor(bool)":
        return np.bool_
    return None


def _cast_for_input(arr: np.ndarray, ort_type: str, use_fp16: bool) -> np.ndarray:
    if use_fp16 and ort_type == "tensor(float16)" and arr.dtype != np.float16:
        return arr.astype(np.float16, copy=False)
    target = _np_dtype_from_ort_type(ort_type)
    if target is not None and arr.dtype != target:
        return arr.astype(target, copy=False)
    return arr



def _build_session(
    onnx_path: str,
    gpu_device_id: int,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.log_severity_level=1
    providers = [
        ("CUDAExecutionProvider", {"device_id": gpu_device_id}),
    ]
    sess = ort.InferenceSession(onnx_path, opts, providers=providers)
    return sess


def _preprocess_frame(frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]]):
    if roi is None:
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        full_t = torch.from_numpy(full_rgb).permute(2, 0, 1).unsqueeze(0).float()
        roi = compute_static_roi(full_t)

    x, y, w, h = roi
    frame_roi = frame_bgr[y : y + h, x : x + w]
    frame_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
        radius = max(0, int(radius) - 4)
        valid = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(valid, (int(cx), int(cy)), radius, 1, -1)
        ellipse_mask = (1 - valid).astype(np.float32)
    else:
        ellipse_mask = np.zeros_like(gray, dtype=np.float32)

    frame = frame_rgb.transpose(2, 0, 1)[None].astype(np.float32)
    ellipse = ellipse_mask[None, None].astype(np.float32)
    return frame, ellipse, roi, frame_roi


def _resize_to_expected(
    frame: np.ndarray,
    ellipse: np.ndarray,
    expected_h: Optional[int],
    expected_w: Optional[int],
):
    if expected_h is None or expected_w is None:
        return frame, ellipse
    h, w = frame.shape[-2:]
    if h == expected_h and w == expected_w:
        return frame, ellipse
    frame_hwc = frame[0].transpose(1, 2, 0)
    ellipse_hw = ellipse[0, 0]
    frame_hwc = cv2.resize(frame_hwc, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    ellipse_hw = cv2.resize(ellipse_hw, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
    frame = frame_hwc.transpose(2, 0, 1)[None].astype(np.float32)
    ellipse = ellipse_hw[None, None].astype(np.float32)
    return frame, ellipse


def _as_numpy(value: Union[np.ndarray, ort.OrtValue]) -> np.ndarray:
    if isinstance(value, ort.OrtValue):
        return value.numpy()
    return value


def _ortvalue_from_input_numpy(
    arr: np.ndarray,
    ort_type: str,
    use_fp16: bool,
    gpu_device_id: int,
) -> ort.OrtValue:
    casted = _cast_for_input(arr, ort_type, use_fp16)
    return ort.OrtValue.ortvalue_from_numpy(casted, "cuda", gpu_device_id)


def _init_state_from_first_frame_iobinding(
    init_sess: ort.InferenceSession,
    frame_np: np.ndarray,
    ellipse_np: np.ndarray,
    use_fp16: bool,
    gpu_device_id: int,
) -> Dict[str, ort.OrtValue]:
    inmeta = {i.name: i for i in init_sess.get_inputs()}
    required = [x.name for x in init_sess.get_inputs()]
    input_ortvalues: Dict[str, ort.OrtValue] = {}
    for name in required:
        if name == "frame":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                frame_np, inmeta[name].type, use_fp16, gpu_device_id
            )
        elif name == "ellipse_mask":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                ellipse_np, inmeta[name].type, use_fp16, gpu_device_id
            )
        else:
            raise RuntimeError(f"Unexpected init input: {name}")

    out_names = [o.name for o in init_sess.get_outputs()]
    io_binding = init_sess.io_binding()
    for name in required:
        io_binding.bind_ortvalue_input(name, input_ortvalues[name])
    for name in out_names:
        io_binding.bind_output(name, "cuda", gpu_device_id)
    io_binding.synchronize_inputs()
    init_sess.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    out_vals = io_binding.get_outputs()

    out_names = [o.name for o in init_sess.get_outputs()]
    out = {name: value for name, value in zip(out_names, out_vals)}
    state = {
        "canvas": out["canvas"],
        "canvas_mask": out["canvas_mask"],
        "H_cum_curr": out["H_cum_curr"],
        "prev_keypoints": out["prev_keypoints"],
        "prev_descriptors": out["prev_descriptors"],
    }
    return state


def _run_step_iobinding(
    step_sess: ort.InferenceSession,
    inmeta: Dict[str, ort.NodeArg],
    out_names: list[str],
    frame_np: np.ndarray,
    ellipse_np: np.ndarray,
    state: Dict[str, ort.OrtValue],
    use_fp16: bool,
    gpu_device_id: int,
) -> Dict[str, ort.OrtValue]:
    input_ortvalues: Dict[str, ort.OrtValue] = {}
    for name in inmeta.keys():
        if name == "frame":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                frame_np, inmeta[name].type, use_fp16, gpu_device_id
            )
        elif name == "ellipse_mask":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                ellipse_np, inmeta[name].type, use_fp16, gpu_device_id
            )
        elif name in state:
            input_ortvalues[name] = state[name]
        else:
            raise RuntimeError(f"Missing required input state: {name}")

    io_binding = step_sess.io_binding()
    for name in inmeta.keys():
        io_binding.bind_ortvalue_input(name, input_ortvalues[name])
    for name in out_names:
        io_binding.bind_output(name, "cuda", gpu_device_id)
    io_binding.synchronize_inputs()
    step_sess.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    out_vals = io_binding.get_outputs()
    return {name: value for name, value in zip(out_names, out_vals)}


def _update_state_from_outputs(state: Dict[str, ort.OrtValue], out: Dict[str, ort.OrtValue]):
    if "canvas_out" in out and "canvas" in state:
        state["canvas"] = out["canvas_out"]
    if "canvas_mask_out" in out and "canvas_mask" in state:
        state["canvas_mask"] = out["canvas_mask_out"]
    if "H_cum_next" in out and "H_cum_curr" in state:
        state["H_cum_curr"] = out["H_cum_next"]
    if "prev_keypoints_next" in out and "prev_keypoints" in state:
        state["prev_keypoints"] = out["prev_keypoints_next"]
    if "prev_descriptors_next" in out and "prev_descriptors" in state:
        state["prev_descriptors"] = out["prev_descriptors_next"]

def _render_overlay(frame_bgr: np.ndarray, seg_map: np.ndarray, alpha: float) -> np.ndarray:
    if seg_map.ndim == 4:
        seg_map = seg_map[0]
    labels = np.argmax(seg_map, axis=0).astype(np.uint8)
    scale = max(1, 255 // max(1, int(labels.max())))
    colored = cv2.applyColorMap((labels * scale).astype(np.uint8), cv2.COLORMAP_TURBO)
    if colored.shape[:2] != frame_bgr.shape[:2]:
        colored = cv2.resize(colored, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(colored, alpha, frame_bgr, 1.0 - alpha, 0.0)

def _canvas_to_bgr(canvas: np.ndarray) -> np.ndarray:
    if canvas.ndim == 4:
        canvas = canvas[0]
    if canvas.shape[0] == 3:
        canvas_hwc = np.transpose(canvas, (1, 2, 0))
    else:
        raise RuntimeError(f"Unexpected canvas shape: {canvas.shape}")
    canvas_hwc = np.clip(canvas_hwc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(canvas_hwc, cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    step_sess = _build_session(
        args.onnx_step,
        args.gpu_device_id,
    )
    init_onnx = args.onnx_init
    if init_onnx is None:
        p = Path(args.onnx_step)
        init_onnx = str(p.with_name(f"{p.stem}_init{p.suffix}"))
    init_sess = _build_session(
        init_onnx,
        args.gpu_device_id,
    )
    print(f"[info] providers(step)={step_sess.get_providers()}")
    print(f"[info] providers(init)={init_sess.get_providers()}")

    inmeta = {i.name: i for i in step_sess.get_inputs()}
    out_names = [o.name for o in step_sess.get_outputs()]
    print(f"[info] inputs={[k for k in inmeta.keys()]}")
    print(f"[info] outputs={out_names}")

    frame_shape = _shape_to_list(inmeta["frame"].shape) if "frame" in inmeta else [None, None, None, None]
    exp_h = frame_shape[2] if len(frame_shape) > 2 else None
    exp_w = frame_shape[3] if len(frame_shape) > 3 else None

    state: Optional[Dict[str, ort.OrtValue]] = None
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    out_path = Path(args.out_video)
    if args.debug:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    total = max(0, args.end_frame - args.start_frame + 1)
    roi = None
    pbar = tqdm(total=total, desc="ONNX")

    frame_idx = args.start_frame
    while frame_idx <= args.end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_np, ellipse_np, roi, frame_roi_bgr = _preprocess_frame(frame_bgr, roi)
        frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)

        if state is None:
            state = _init_state_from_first_frame_iobinding(
                init_sess=init_sess,
                frame_np=frame_np,
                ellipse_np=ellipse_np,
                use_fp16=args.fp16,
                gpu_device_id=args.gpu_device_id,
            )

        out = _run_step_iobinding(
            step_sess=step_sess,
            inmeta=inmeta,
            out_names=out_names,
            frame_np=frame_np,
            ellipse_np=ellipse_np,
            state=state,
            use_fp16=args.fp16,
            gpu_device_id=args.gpu_device_id,
        )
        _update_state_from_outputs(state, out)

        seg_map_ort = out.get("seg_map")
        seg_map = _as_numpy(seg_map_ort) if seg_map_ort is not None else None
        if seg_map is None:
            raise RuntimeError("seg_map was not found in ONNX outputs.")

        if args.debug:
            if frame_roi_bgr.shape[:2] != seg_map.shape[-2:]:
                frame_vis = cv2.resize(
                    frame_roi_bgr,
                    (seg_map.shape[-1], seg_map.shape[-2]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_vis = frame_roi_bgr
            overlay = _render_overlay(frame_vis, seg_map, alpha=args.overlay_alpha)
            canvas_bgr = _canvas_to_bgr(_as_numpy(state["canvas"]))
            if canvas_bgr.shape[0] != overlay.shape[0]:
                new_w = max(1, int(round(canvas_bgr.shape[1] * overlay.shape[0] / canvas_bgr.shape[0])))
                canvas_bgr = cv2.resize(canvas_bgr, (new_w, overlay.shape[0]), interpolation=cv2.INTER_LINEAR)
            vis = np.hstack([overlay, canvas_bgr])

            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (vis.shape[1], vis.shape[0]), True)
            writer.write(vis)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    if args.debug:
        print(f"[done] saved: {out_path}")
    else:
        print("[done] debug video disabled (--debug not set).")


if __name__ == "__main__":
    main()
