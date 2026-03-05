#!/usr/bin/env python3
"""Benchmark ONNX full pipeline (stitch + segmentation) as in run_onnx.py.

Measures FPS and latency (p95) for the same inference path as run_onnx.py.

Usage:
    python bench_onnx.py --start 10 --end 990 --name stitchunet.json
    python bench_onnx.py --start 10 --end 990 --name stitchunet.json --fp16
    python bench_onnx.py --start 10 --end 990 --name stitchunet.json --warmup 20
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from stitch_seg import compute_static_roi


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ONNX full pipeline (stitch + segmentation) as in run_onnx.py.")
    p.add_argument("--onnx-file", type=str, default="try_cuda.onnx",
                    help="Path to ONNX model.")
    p.add_argument("--onnx-init", type=str, default=None,
                    help="Path to first_frame ONNX. Defaults to <onnx-file stem>_init.onnx")
    p.add_argument("--video", type=str, default="video01.mp4", help="Input video path.")
    p.add_argument("--start", type=int, default=100,
                    help="First frame to *measure* (after warmup).")
    p.add_argument("--end", type=int, default=10000,
                    help="Last frame to measure (inclusive).")
    p.add_argument("--warmup", type=int, default=100,
                    help="Number of warmup frames before --start (not included in stats).")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--fp16", action="store_true",
                    help="Load .fp16.onnx models.")
    p.add_argument("--name", type=str, default="upernet.json",
                    help="Output JSON path for benchmark results.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (aligned with run_onnx.py)
# ---------------------------------------------------------------------------

def _shape_to_list(shape):
    return [d if isinstance(d, int) else None for d in shape]


def _np_dtype_from_ort_type(ort_type: str):
    _map = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return _map.get(ort_type)


def _cast_for_input(arr: np.ndarray, ort_type: str, use_fp16: bool) -> np.ndarray:
    if use_fp16 and ort_type == "tensor(float16)" and arr.dtype != np.float16:
        return arr.astype(np.float16, copy=False)
    target = _np_dtype_from_ort_type(ort_type)
    if target is not None and arr.dtype != target:
        return arr.astype(target, copy=False)
    return arr


def _build_session(onnx_path: str, gpu_device_id: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_profiling = False
    opts.log_severity_level = 2  # quieter for benchmarks
    providers = [
        ("CUDAExecutionProvider", {"device_id": gpu_device_id}),
    ]
    return ort.InferenceSession(onnx_path, opts, providers=providers)


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
    return frame, ellipse, roi


def _resize_to_expected(frame, ellipse, expected_h, expected_w):
    if expected_h is None or expected_w is None:
        return frame, ellipse
    h, w = frame.shape[-2:]
    if h == expected_h and w == expected_w:
        return frame, ellipse
    frame_hwc = frame[0].transpose(1, 2, 0)
    ellipse_hw = ellipse[0, 0]
    frame_hwc = cv2.resize(frame_hwc, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    ellipse_hw = cv2.resize(ellipse_hw, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
    return frame_hwc.transpose(2, 0, 1)[None].astype(np.float32), ellipse_hw[None, None].astype(np.float32)


def _as_numpy(value: Union[np.ndarray, ort.OrtValue]) -> np.ndarray:
    if isinstance(value, ort.OrtValue):
        return value.numpy()
    return value


def _ortvalue_from_input_numpy(arr, ort_type, use_fp16, gpu_device_id):
    casted = _cast_for_input(arr, ort_type, use_fp16)
    return ort.OrtValue.ortvalue_from_numpy(casted, "cuda", gpu_device_id)


# ---------------------------------------------------------------------------
# IOBinding inference wrappers
# ---------------------------------------------------------------------------

def _init_state(init_sess, frame_np, ellipse_np, use_fp16, gpu_device_id):
    """Run first_frame ONNX; returns (state, seg_map_ort). seg_map_ort is None for stitch-only models."""
    inmeta = {i.name: i for i in init_sess.get_inputs()}
    required = [x.name for x in init_sess.get_inputs()]
    input_ortvalues = {}
    for name in required:
        if name == "frame":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                frame_np, inmeta[name].type, use_fp16, gpu_device_id)
        elif name == "ellipse_mask":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                ellipse_np, inmeta[name].type, use_fp16, gpu_device_id)
        else:
            raise RuntimeError(f"Unexpected init input: {name}")

    out_names = [o.name for o in init_sess.get_outputs()]
    io = init_sess.io_binding()
    for name in required:
        io.bind_ortvalue_input(name, input_ortvalues[name])
    for name in out_names:
        io.bind_output(name, "cuda", gpu_device_id)
    io.synchronize_inputs()
    init_sess.run_with_iobinding(io)
    io.synchronize_outputs()
    out_vals = io.get_outputs()
    out = dict(zip(out_names, out_vals))
    state = {
        "canvas": out["canvas"],
        "canvas_mask": out["canvas_mask"],
        "H_cum_curr": out["H_cum_curr"],
        "prev_keypoints": out["prev_keypoints"],
        "prev_descriptors": out["prev_descriptors"],
    }
    seg_map_ort = out.get("seg_map")
    return state, seg_map_ort


def _run_step(step_sess, inmeta, out_names, frame_np, ellipse_np,
              state, use_fp16, gpu_device_id):
    input_ortvalues = {}
    for name in inmeta.keys():
        if name == "frame":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                frame_np, inmeta[name].type, use_fp16, gpu_device_id)
        elif name == "ellipse_mask":
            input_ortvalues[name] = _ortvalue_from_input_numpy(
                ellipse_np, inmeta[name].type, use_fp16, gpu_device_id)
        elif name in state:
            input_ortvalues[name] = state[name]
        else:
            raise RuntimeError(f"Missing required input state: {name}")

    io = step_sess.io_binding()
    for name in inmeta.keys():
        io.bind_ortvalue_input(name, input_ortvalues[name])
    for name in out_names:
        io.bind_output(name, "cuda", gpu_device_id)
    io.synchronize_inputs()
    step_sess.run_with_iobinding(io)
    io.synchronize_outputs()
    out_vals = io.get_outputs()
    return dict(zip(out_names, out_vals))


def _update_state_from_outputs(state: Dict, out: Dict):
    """Update state from step outputs (same as run_onnx.py)."""
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


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve paths
    if args.onnx_init is None:
        p = Path(args.onnx_file)
        args.onnx_init = str(p.with_name(f"{p.stem}_init{p.suffix}"))
    if args.fp16:
        for attr in ("onnx_file", "onnx_init"):
            p = Path(getattr(args, attr))
            if ".fp16." not in p.name:
                setattr(args, attr, str(p.with_suffix(".fp16.onnx")))

    print(f"[bench] step={args.onnx_file}")
    print(f"[bench] init={args.onnx_init}")
    print(f"[bench] video={args.video}")
    print(f"[bench] frames: warmup={args.warmup}, measure=[{args.start}, {args.end}]")

    step_sess = _build_session(args.onnx_file, args.gpu_device_id)
    init_sess = _build_session(args.onnx_init, args.gpu_device_id)
    print(f"[bench] providers(step)={step_sess.get_providers()}")

    inmeta = {i.name: i for i in step_sess.get_inputs()}
    out_names = [o.name for o in step_sess.get_outputs()]
    frame_shape = _shape_to_list(inmeta["frame"].shape) if "frame" in inmeta else [None]*4
    exp_h = frame_shape[2] if len(frame_shape) > 2 else None
    exp_w = frame_shape[3] if len(frame_shape) > 3 else None

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Seek to warmup start: warmup frames precede args.start
    warmup_start = max(0, args.start - args.warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

    state = None
    roi = None
    latencies_ms = []  # per-frame latencies for measured frames
    reset_count = 0
    total_measure = args.end - args.start + 1
    pbar = tqdm(total=total_measure, desc="Bench")

    frame_idx = warmup_start
    while frame_idx <= args.end:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Init on first frame (preprocess before timing)
        if state is None:
            frame_np, ellipse_np, roi = _preprocess_frame(frame_bgr, roi)
            frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)
            state, _ = _init_state(init_sess, frame_np, ellipse_np,
                                   args.fp16, args.gpu_device_id)
            frame_idx += 1
            if frame_idx > args.start:
                pbar.update(1)
            continue

        # Determine whether this frame is measured
        is_measured = frame_idx >= args.start

        frame_np, ellipse_np, roi = _preprocess_frame(frame_bgr, roi)
        frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)

        # --- Timed region: inference only ---
        if is_measured:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        out = _run_step(step_sess, inmeta, out_names,
                        frame_np, ellipse_np, state,
                        args.fp16, args.gpu_device_id)

        if is_measured:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            pbar.update(1)

        # Check needs_reset (outside timed region)
        needs_reset_ort = out.get("needs_reset")
        needs_reset = False
        if needs_reset_ort is not None:
            nr = _as_numpy(needs_reset_ort)
            needs_reset = bool(nr.item()) if nr.size == 1 else bool(nr.any())

        if needs_reset:
            state, _ = _init_state(init_sess, frame_np, ellipse_np,
                                  args.fp16, args.gpu_device_id)
            reset_count += 1
        else:
            _update_state_from_outputs(state, out)

        frame_idx += 1

    pbar.close()
    cap.release()

    # --- Compute statistics ---
    if not latencies_ms:
        print("[bench] No frames measured!")
        return

    lat = np.array(latencies_ms)
    n = len(lat)
    mean_ms = float(np.mean(lat))
    median_ms = float(np.median(lat))
    std_ms = float(np.std(lat))
    min_ms = float(np.min(lat))
    max_ms = float(np.max(lat))
    p50_ms = float(np.percentile(lat, 50))
    p90_ms = float(np.percentile(lat, 90))
    p95_ms = float(np.percentile(lat, 95))
    p99_ms = float(np.percentile(lat, 99))
    fps_mean = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    fps_p95 = 1000.0 / p95_ms if p95_ms > 0 else 0.0

    results = {
        "model": {
            "onnx_file": args.onnx_file,
            "onnx_init": args.onnx_init,
            "fp16": args.fp16,
            "providers": step_sess.get_providers(),
        },
        "benchmark": {
            "video": args.video,
            "warmup_frames": args.warmup,
            "start_frame": args.start,
            "end_frame": args.end,
            "measured_frames": n,
            "reset_count": reset_count,
        },
        "latency_ms": {
            "mean": round(mean_ms, 3),
            "median": round(median_ms, 3),
            "std": round(std_ms, 3),
            "min": round(min_ms, 3),
            "max": round(max_ms, 3),
            "p50": round(p50_ms, 3),
            "p90": round(p90_ms, 3),
            "p95": round(p95_ms, 3),
            "p99": round(p99_ms, 3),
        },
        "fps": {
            "mean": round(fps_mean, 3),
            "at_p95_latency": round(fps_p95, 3),
        },
        "per_frame_ms": [round(float(v), 3) for v in lat],
    }

    # --- Print summary ---
    print()
    print("=" * 60)
    print(f"  Benchmark Results: {args.name}")
    print("=" * 60)
    print(f"  Measured frames : {n}")
    print(f"  Resets          : {reset_count}")
    print(f"  FPS (mean)      : {fps_mean:.2f}")
    print(f"  FPS (at p95 lat): {fps_p95:.2f}")
    print(f"  Latency mean    : {mean_ms:.3f} ms")
    print(f"  Latency median  : {median_ms:.3f} ms")
    print(f"  Latency p90     : {p90_ms:.3f} ms")
    print(f"  Latency p95     : {p95_ms:.3f} ms")
    print(f"  Latency p99     : {p99_ms:.3f} ms")
    print(f"  Latency min/max : {min_ms:.3f} / {max_ms:.3f} ms")
    print("=" * 60)

    # --- Save JSON ---
    out_path = Path(args.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
