#!/usr/bin/env python3
"""Run exported stitch ONNX models with TensorRT Execution Provider."""
import argparse
from pathlib import Path

import cv2
import onnxruntime as ort
from tqdm import tqdm

from run_onnx import (
    _cast_for_input,
    _canvas_to_bgr,
    _init_state_from_first_frame,
    _preprocess_frame,
    _render_overlay,
    _resize_to_expected,
    _shape_to_list,
    _update_state_from_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ONNX stitch inferencer with TensorRT EP.")
    p.add_argument("--onnx-step", type=str, default="try.onnx", help="Path to step/update ONNX.")
    p.add_argument(
        "--onnx-init",
        type=str,
        default=None,
        help="Path to first_frame ONNX. Defaults to <onnx-step stem>_init.onnx",
    )
    p.add_argument("--video", type=str, default="video01.mp4", help="Input video path.")
    p.add_argument("--start-frame", type=int, default=28000)
    p.add_argument("--end-frame", type=int, default=29000)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--trt-engine-cache-enable", action="store_true")
    p.add_argument("--trt-engine-cache-path", type=str, default=".trt_cache")
    p.add_argument("--trt-fp16", action="store_true")
    p.add_argument("--trt-int8", action="store_true")
    p.add_argument("--fp16", action="store_true", help="Feed FP16 for float16 model inputs/states.")
    p.add_argument("--out-video", type=str, default="trt_overlay.mp4")
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument("--debug", action="store_true", help="Render and save debug video.")
    return p.parse_args()


def _build_trt_session(
    onnx_path: str,
    gpu_device_id: int,
    trt_engine_cache_enable: bool,
    trt_engine_cache_path: str,
    trt_fp16: bool,
    trt_int8: bool,
) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in available:
        raise RuntimeError(f"TensorrtExecutionProvider is unavailable. available={available}")

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": gpu_device_id,
                "trt_engine_cache_enable": trt_engine_cache_enable,
                "trt_engine_cache_path": trt_engine_cache_path,
                "trt_fp16_enable": trt_fp16,
                "trt_int8_enable": trt_int8,
            },
        ),
        ("CUDAExecutionProvider", {"device_id": gpu_device_id}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    sess_providers = sess.get_providers()
    if "TensorrtExecutionProvider" not in sess_providers:
        raise RuntimeError(
            "TensorRT EP was not enabled at runtime. "
            f"enabled={sess_providers}. "
            "Check libnvinfer/libnvonnxparser/libcudnn versions and LD_LIBRARY_PATH."
        )
    return sess


def main():
    args = parse_args()
    if args.trt_engine_cache_enable:
        Path(args.trt_engine_cache_path).mkdir(parents=True, exist_ok=True)

    init_onnx = args.onnx_init
    if init_onnx is None:
        p = Path(args.onnx_step)
        init_onnx = str(p.with_name(f"{p.stem}_init{p.suffix}"))

    step_sess = _build_trt_session(
        args.onnx_step,
        args.gpu_device_id,
        args.trt_engine_cache_enable,
        args.trt_engine_cache_path,
        args.trt_fp16,
        args.trt_int8,
    )
    init_sess = _build_trt_session(
        init_onnx,
        args.gpu_device_id,
        args.trt_engine_cache_enable,
        args.trt_engine_cache_path,
        args.trt_fp16,
        args.trt_int8,
    )

    print(f"[info] providers(step)={step_sess.get_providers()}")
    print(f"[info] providers(init)={init_sess.get_providers()}")

    inmeta = {i.name: i for i in step_sess.get_inputs()}
    inmeta_init = {i.name: i for i in init_sess.get_inputs()}
    out_names = [o.name for o in step_sess.get_outputs()]

    frame_shape = _shape_to_list(inmeta["frame"].shape) if "frame" in inmeta else [None, None, None, None]
    exp_h = frame_shape[2] if len(frame_shape) > 2 else None
    exp_w = frame_shape[3] if len(frame_shape) > 3 else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    out_path = Path(args.out_video)
    if args.debug:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    state = None
    roi = None
    total = max(0, args.end_frame - args.start_frame + 1)
    pbar = tqdm(total=total, desc="TRT")

    frame_idx = args.start_frame
    while frame_idx <= args.end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_np, ellipse_np, roi, frame_roi_bgr = _preprocess_frame(frame_bgr, roi)
        frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)

        if state is None:
            state = _init_state_from_first_frame(
                init_sess=init_sess,
                frame_np=_cast_for_input(frame_np, inmeta_init["frame"].type, args.fp16),
                ellipse_np=_cast_for_input(ellipse_np, inmeta_init["ellipse_mask"].type, args.fp16),
            )

        feeds = {}
        for name in inmeta.keys():
            if name == "frame":
                feeds[name] = _cast_for_input(frame_np, inmeta[name].type, args.fp16)
            elif name == "ellipse_mask":
                feeds[name] = _cast_for_input(ellipse_np, inmeta[name].type, args.fp16)
            elif name in state:
                feeds[name] = _cast_for_input(state[name], inmeta[name].type, args.fp16)
            else:
                raise RuntimeError(f"Missing required input state: {name}")

        out_vals = step_sess.run(None, feeds)
        out = {name: value for name, value in zip(out_names, out_vals)}
        _update_state_from_outputs(state, out)

        if args.debug:
            seg_map = out["seg_map"]
            if frame_roi_bgr.shape[:2] != seg_map.shape[-2:]:
                frame_vis = cv2.resize(
                    frame_roi_bgr,
                    (seg_map.shape[-1], seg_map.shape[-2]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_vis = frame_roi_bgr
            overlay = _render_overlay(frame_vis, seg_map, alpha=args.overlay_alpha)
            canvas_bgr = _canvas_to_bgr(state["canvas"])
            if canvas_bgr.shape[0] != overlay.shape[0]:
                new_w = max(1, int(round(canvas_bgr.shape[1] * overlay.shape[0] / canvas_bgr.shape[0])))
                canvas_bgr = cv2.resize(canvas_bgr, (new_w, overlay.shape[0]), interpolation=cv2.INTER_LINEAR)
            vis = cv2.hconcat([overlay, canvas_bgr])

            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (vis.shape[1], vis.shape[0]),
                    True,
                )
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
