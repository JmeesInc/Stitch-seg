#!/usr/bin/env python3
"""Run stitch-only TRT .engine (or ONNX + TRT EP) with IOBinding for zero-copy GPU execution.

Stitching only — no segmentation.  Debug output: current frame | canvas side-by-side.

Usage (pre-built engines):
    python3 run_tensorrt_only.py \
        --engine-step trt_engine_only/only.engine \
        --video video01.mp4 --start-frame 0 --end-frame 3333 --debug

Usage (ONNX + TRT EP fallback):
    python3 run_tensorrt_only.py \
        --onnx-step only.onnx --trt \
        --video video01.mp4 --debug
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from run_tensorrt import (
    _as_numpy,
    _build_session,
    _build_trt_session,
    _ortvalue_from_numpy,
)
from run_onnx_only import (
    _canvas_to_bgr,
    _frame_np_to_bgr,
    _preprocess_frame,
    _resize_to_expected,
    _shape_to_list,
    _update_state_from_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run stitch-only TRT inferencer with IOBinding."
    )
    p.add_argument("--engine-step", type=str, default="trt_engine_only/only.engine",
                    help="Pre-built TRT .engine for the step model.")
    p.add_argument("--engine-init", type=str, default=None,
                    help="Pre-built TRT .engine for the init model. "
                         "Defaults to <engine-step stem>_init.engine.")
    p.add_argument("--onnx-step", type=str, default="only.onnx",
                    help="Step ONNX path (used when --engine-step is not set).")
    p.add_argument("--onnx-init", type=str, default=None,
                    help="Init ONNX path. Defaults to <onnx-step stem>_init.onnx")
    p.add_argument("--video", type=str, default="video01.mp4", help="Input video path.")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=3333)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--fp16", action="store_true",
                    help="Load .fp16.onnx models (ONNX path only).")
    p.add_argument("--out-video", type=str, default="trt_only.mp4")
    p.add_argument("--debug", action="store_true",
                    help="Render and save debug video (frame | canvas).")
    p.add_argument("--trt", action="store_true",
                    help="Use TensorRT EP for ONNX sessions (--engine-step takes precedence).")
    # Benchmark options (mutually exclusive with --debug)
    p.add_argument("--bench", action="store_true",
                    help="Run in benchmark mode: measure per-frame step latency and save JSON stats. "
                         "Mutually exclusive with --debug.")
    p.add_argument("--warmup", type=int, default=100,
                    help="Warmup frames before --start-frame (bench mode only, not included in stats).")
    p.add_argument("--bench-name", type=str, default="trt_only_bench.json",
                    help="Output JSON path for benchmark results (bench mode only).")
    return p.parse_args()


def _run_init_iobinding(
    init_sess,
    frame_np: np.ndarray,
    ellipse_np: np.ndarray,
    use_fp16: bool,
    gpu_id: int,
) -> Dict[str, ort.OrtValue]:
    """Run first_frame session with IOBinding; returns state dict of OrtValues on GPU."""
    inmeta = {i.name: i for i in init_sess.get_inputs()}
    input_ortvals: Dict[str, ort.OrtValue] = {}
    for name in inmeta:
        if name == "frame":
            input_ortvals[name] = _ortvalue_from_numpy(frame_np, inmeta[name].type, use_fp16, gpu_id)
        elif name == "ellipse_mask":
            input_ortvals[name] = _ortvalue_from_numpy(ellipse_np, inmeta[name].type, use_fp16, gpu_id)
        else:
            raise RuntimeError(f"Unexpected init input: {name}")

    out_names = [o.name for o in init_sess.get_outputs()]
    iob = init_sess.io_binding()
    for name in inmeta:
        iob.bind_ortvalue_input(name, input_ortvals[name])
    for name in out_names:
        iob.bind_output(name, "cuda", gpu_id)
    iob.synchronize_inputs()
    init_sess.run_with_iobinding(iob)
    iob.synchronize_outputs()

    out = dict(zip(out_names, iob.get_outputs()))
    return {
        "canvas":           out["canvas"],
        "canvas_mask":      out["canvas_mask"],
        "H_cum_curr":       out["H_cum_curr"],
        "prev_keypoints":   out["prev_keypoints"],
        "prev_descriptors": out["prev_descriptors"],
    }


def _run_step_iobinding(
    step_sess,
    inmeta: Dict,
    out_names: list,
    frame_np: np.ndarray,
    ellipse_np: np.ndarray,
    state: Dict[str, ort.OrtValue],
    use_fp16: bool,
    gpu_id: int,
) -> Dict[str, ort.OrtValue]:
    """Run step session with IOBinding; state OrtValues stay on GPU (zero-copy)."""
    input_ortvals: Dict[str, ort.OrtValue] = {}
    for name in inmeta:
        if name == "frame":
            input_ortvals[name] = _ortvalue_from_numpy(frame_np, inmeta[name].type, use_fp16, gpu_id)
        elif name == "ellipse_mask":
            input_ortvals[name] = _ortvalue_from_numpy(ellipse_np, inmeta[name].type, use_fp16, gpu_id)
        elif name in state:
            input_ortvals[name] = state[name]  # already on GPU
        else:
            raise RuntimeError(f"Missing required input state: {name}")

    iob = step_sess.io_binding()
    for name in inmeta:
        iob.bind_ortvalue_input(name, input_ortvals[name])
    for name in out_names:
        iob.bind_output(name, "cuda", gpu_id)
    iob.synchronize_inputs()
    step_sess.run_with_iobinding(iob)
    iob.synchronize_outputs()
    return dict(zip(out_names, iob.get_outputs()))


def main():
    args = parse_args()

    if args.bench and args.debug:
        raise SystemExit("[error] --bench and --debug are mutually exclusive.")

    if args.onnx_init is None:
        p = Path(args.onnx_step)
        args.onnx_init = str(p.with_name(f"{p.stem}_init{p.suffix}"))

    # Derive engine-init from engine-step when only step is given
    if args.engine_step and args.engine_init is None:
        ep = Path(args.engine_step)
        args.engine_init = str(ep.with_name(f"{ep.stem}_init{ep.suffix}"))

    if args.fp16:
        for attr in ("onnx_step", "onnx_init"):
            p = Path(getattr(args, attr))
            if ".fp16." not in p.name:
                setattr(args, attr, str(p.with_suffix(".fp16.onnx")))

    if args.engine_step:
        # Use pre-built TRT .engine files directly (no ONNX loading / TRT EP compilation)
        print(f"[info] engine-step={args.engine_step}, engine-init={args.engine_init}")
        step_sess = _build_trt_session(args.engine_step, args.gpu_device_id)
        init_sess = _build_trt_session(args.engine_init, args.gpu_device_id)
    else:
        # ONNX path: optionally use TRT EP (--trt)
        print(f"[info] step={args.onnx_step}, init={args.onnx_init}")
        step_sess = _build_session(args.onnx_step, args.gpu_device_id, use_trt=args.trt)
        init_sess = _build_session(args.onnx_init, args.gpu_device_id, use_trt=args.trt)
    print(f"[info] providers(step)={step_sess.get_providers()}")
    print(f"[info] providers(init)={init_sess.get_providers()}")

    inmeta = {i.name: i for i in step_sess.get_inputs()}
    out_names = [o.name for o in step_sess.get_outputs()]

    frame_shape = _shape_to_list(inmeta["frame"].shape) if "frame" in inmeta else [None] * 4
    exp_h = frame_shape[2] if len(frame_shape) > 2 else None
    exp_w = frame_shape[3] if len(frame_shape) > 3 else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # -----------------------------------------------------------------------
    # Benchmark mode
    # -----------------------------------------------------------------------
    if args.bench:
        warmup_start = max(0, args.start_frame - args.warmup)
        cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

        state: Optional[Dict[str, ort.OrtValue]] = None
        roi = None
        latencies_ms: List[float] = []
        reset_count = 0
        total_measure = max(0, args.end_frame - args.start_frame + 1)
        desc = "bench-engine-only" if args.engine_step else ("bench-TRT-only" if args.trt else "bench-CUDA-only")
        pbar = tqdm(total=total_measure, desc=desc)

        frame_idx = warmup_start
        while frame_idx <= args.end_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_np, ellipse_np, roi, _ = _preprocess_frame(frame_bgr, roi)
            frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)

            # First frame: initialise state, skip timing
            if state is None:
                state = _run_init_iobinding(
                    init_sess, frame_np, ellipse_np, args.fp16, args.gpu_device_id,
                )
                frame_idx += 1
                if frame_idx > args.start_frame:
                    pbar.update(1)
                continue

            is_measured = frame_idx >= args.start_frame

            if is_measured:
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            out = _run_step_iobinding(
                step_sess, inmeta, out_names,
                frame_np, ellipse_np, state,
                args.fp16, args.gpu_device_id,
            )

            if is_measured:
                torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                pbar.update(1)

            needs_reset_ort = out.get("needs_reset")
            needs_reset = False
            if needs_reset_ort is not None:
                nr = _as_numpy(needs_reset_ort)
                needs_reset = bool(nr.item()) if nr.size == 1 else bool(nr.any())

            if needs_reset:
                state = _run_init_iobinding(
                    init_sess, frame_np, ellipse_np, args.fp16, args.gpu_device_id,
                )
                reset_count += 1
            else:
                _update_state_from_outputs(state, out)

            frame_idx += 1

        pbar.close()
        cap.release()

        if not latencies_ms:
            print("[bench] No frames measured!")
            return

        lat = np.array(latencies_ms)
        n         = len(lat)
        mean_ms   = float(np.mean(lat))
        median_ms = float(np.median(lat))
        std_ms    = float(np.std(lat))
        min_ms    = float(np.min(lat))
        max_ms    = float(np.max(lat))
        p50_ms    = float(np.percentile(lat, 50))
        p90_ms    = float(np.percentile(lat, 90))
        p95_ms    = float(np.percentile(lat, 95))
        p99_ms    = float(np.percentile(lat, 99))
        fps_mean  = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        fps_p95   = 1000.0 / p95_ms  if p95_ms  > 0 else 0.0

        results = {
            "model": {
                "step":      args.engine_step or args.onnx_step,
                "init":      args.engine_init or args.onnx_init,
                "fp16":      args.fp16,
                "providers": step_sess.get_providers(),
            },
            "benchmark": {
                "video":           args.video,
                "warmup_frames":   args.warmup,
                "start_frame":     args.start_frame,
                "end_frame":       args.end_frame,
                "measured_frames": n,
                "reset_count":     reset_count,
            },
            "latency_ms": {
                "mean":   round(mean_ms,   3),
                "median": round(median_ms, 3),
                "std":    round(std_ms,    3),
                "min":    round(min_ms,    3),
                "max":    round(max_ms,    3),
                "p50":    round(p50_ms,    3),
                "p90":    round(p90_ms,    3),
                "p95":    round(p95_ms,    3),
                "p99":    round(p99_ms,    3),
            },
            "fps": {
                "mean":           round(fps_mean, 3),
                "at_p95_latency": round(fps_p95,  3),
            },
            "per_frame_ms": [round(float(v), 3) for v in lat],
        }

        print()
        print("=" * 60)
        print(f"  Benchmark: {args.bench_name}")
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

        bench_path = Path(args.bench_name)
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bench_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {bench_path}")
        return

    # -----------------------------------------------------------------------
    # Normal (inference / debug-video) mode
    # -----------------------------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    out_path = Path(args.out_video)
    if args.debug:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    vis_size = None

    state: Optional[Dict[str, ort.OrtValue]] = None
    roi = None
    total = max(0, args.end_frame - args.start_frame + 1)
    desc = "engine-only" if args.engine_step else ("TRT-only" if args.trt else "CUDA-only")
    pbar = tqdm(total=total, desc=desc)

    frame_idx = args.start_frame
    while frame_idx <= args.end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_np, ellipse_np, roi, _ = _preprocess_frame(frame_bgr, roi)
        frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)

        if state is None:
            state = _run_init_iobinding(
                init_sess, frame_np, ellipse_np, args.fp16, args.gpu_device_id,
            )

        out = _run_step_iobinding(
            step_sess, inmeta, out_names,
            frame_np, ellipse_np, state,
            args.fp16, args.gpu_device_id,
        )

        needs_reset_ort = out.get("needs_reset")
        needs_reset = False
        if needs_reset_ort is not None:
            nr = _as_numpy(needs_reset_ort)
            needs_reset = bool(nr.item()) if nr.size == 1 else bool(nr.any())

        if needs_reset:
            state = _run_init_iobinding(
                init_sess, frame_np, ellipse_np, args.fp16, args.gpu_device_id,
            )
        else:
            _update_state_from_outputs(state, out)

        if args.debug:
            frame_vis = _frame_np_to_bgr(frame_np)
            canvas_bgr = _canvas_to_bgr(_as_numpy(state["canvas"]))

            if canvas_bgr.shape[0] != frame_vis.shape[0]:
                new_w = max(1, int(round(
                    canvas_bgr.shape[1] * frame_vis.shape[0] / canvas_bgr.shape[0]
                )))
                canvas_bgr = cv2.resize(canvas_bgr, (new_w, frame_vis.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            vis = np.hstack([frame_vis, canvas_bgr])

            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                vis_size = (vis.shape[1], vis.shape[0])
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    vis_size,
                    True,
                )
            else:
                vis = cv2.resize(vis, vis_size, interpolation=cv2.INTER_LINEAR)
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
