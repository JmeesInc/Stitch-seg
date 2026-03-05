#!/usr/bin/env python3
"""Benchmark seg-only ONNX model (no stitch pipeline).

Measures FPS and latency for the segmentation model alone.

Usage:
    python bench_onnx_seg.py --video video01.mp4 --start 100 --end 1000 --name seg.json
    python bench_onnx_seg.py --video video01.mp4 --fp16 --name seg_fp16.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

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
    p = argparse.ArgumentParser(description="Benchmark seg-only ONNX model.")
    p.add_argument("--onnx-file", type=str, default="onnx/seg.onnx",
                   help="Path to seg ONNX model.")
    p.add_argument("--video", type=str, default="video01.mp4", help="Input video path.")
    p.add_argument("--start", type=int, default=100,
                   help="First frame to measure (after warmup).")
    p.add_argument("--end", type=int, default=10000,
                   help="Last frame to measure (inclusive).")
    p.add_argument("--warmup", type=int, default=100,
                   help="Number of warmup frames before --start.")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--fp16", action="store_true",
                   help="Load .fp16.onnx model.")
    p.add_argument("--name", type=str, default="seg.json",
                   help="Output JSON path for benchmark results.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_session(onnx_path: str, gpu_device_id: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_profiling = False
    opts.log_severity_level = 2
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
    frame = frame_rgb.transpose(2, 0, 1)[None].astype(np.float32)
    return frame, roi


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.fp16:
        p = Path(args.onnx_file)
        if ".fp16." not in p.name:
            args.onnx_file = str(p.with_suffix(".fp16.onnx"))

    print(f"[bench_seg] model={args.onnx_file}")
    print(f"[bench_seg] video={args.video}")
    print(f"[bench_seg] frames: warmup={args.warmup}, measure=[{args.start}, {args.end}]")

    sess = _build_session(args.onnx_file, args.gpu_device_id)
    print(f"[bench_seg] providers={sess.get_providers()}")

    inmeta = {i.name: i for i in sess.get_inputs()}
    out_names = [o.name for o in sess.get_outputs()]
    input_name = list(inmeta.keys())[0]  # "image"
    input_type = inmeta[input_name].type
    use_fp16 = args.fp16

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    warmup_start = max(0, args.start - args.warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

    roi = None
    latencies_ms = []
    total_measure = args.end - args.start + 1
    pbar = tqdm(total=total_measure, desc="Bench seg")

    frame_idx = warmup_start
    while frame_idx <= args.end:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_np, roi = _preprocess_frame(frame_bgr, roi)

        # Cast to model input dtype
        target_dtype = _np_dtype_from_ort_type(input_type)
        if target_dtype is not None and frame_np.dtype != target_dtype:
            frame_np_cast = frame_np.astype(target_dtype, copy=False)
        else:
            frame_np_cast = frame_np

        is_measured = frame_idx >= args.start

        # IOBinding for GPU-resident inference
        ort_input = ort.OrtValue.ortvalue_from_numpy(frame_np_cast, "cuda", args.gpu_device_id)
        io = sess.io_binding()
        io.bind_ortvalue_input(input_name, ort_input)
        for name in out_names:
            io.bind_output(name, "cuda", args.gpu_device_id)
        io.synchronize_inputs()

        if is_measured:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        sess.run_with_iobinding(io)

        if is_measured:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            pbar.update(1)

        io.synchronize_outputs()
        frame_idx += 1

    pbar.close()
    cap.release()

    # --- Statistics ---
    if not latencies_ms:
        print("[bench_seg] No frames measured!")
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
            "fp16": args.fp16,
            "providers": sess.get_providers(),
        },
        "benchmark": {
            "video": args.video,
            "warmup_frames": args.warmup,
            "start_frame": args.start,
            "end_frame": args.end,
            "measured_frames": n,
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

    print()
    print("=" * 60)
    print(f"  Benchmark Results: {args.name}")
    print("=" * 60)
    print(f"  Measured frames : {n}")
    print(f"  FPS (mean)      : {fps_mean:.2f}")
    print(f"  FPS (at p95 lat): {fps_p95:.2f}")
    print(f"  Latency mean    : {mean_ms:.3f} ms")
    print(f"  Latency median  : {median_ms:.3f} ms")
    print(f"  Latency p90     : {p90_ms:.3f} ms")
    print(f"  Latency p95     : {p95_ms:.3f} ms")
    print(f"  Latency p99     : {p99_ms:.3f} ms")
    print(f"  Latency min/max : {min_ms:.3f} / {max_ms:.3f} ms")
    print("=" * 60)

    out_path = Path(args.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
