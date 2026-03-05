#!/usr/bin/env python3
"""Run exported stitch ONNX models with TensorRT EP + IOBinding for zero-copy GPU execution.

TensorRT note:
  The step model contains homography estimation (DLT solver via Gauss-Jordan
  matrix inverse).  TensorRT's kernel fusion incorrectly handles the Gauss-Jordan
  elimination Div chain, producing NaN.  The workaround is to expose the 8
  pivot-division intermediate tensors (Div_16..Div_23) as model outputs, which
  forces TRT to break its fusion around the matrix inverse and compute correctly.
  Both init and step models run on TRT EP when --trt is specified.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort
import torch
from tqdm import tqdm

from run_onnx import (
    _canvas_to_bgr,
    _cast_for_input,
    _preprocess_frame,
    _render_overlay,
    _resize_to_expected,
    _shape_to_list,
    _update_state_from_outputs,
)

# Gauss-Jordan pivot-division outputs whose exposure forces TRT to break its
# kernel fusion around the 8x8 matrix inverse in the DLT homography solver.
_TRT_FUSION_BREAK_OUTPUTS: List[str] = [
    f"/Div_{i}_output_0" for i in range(16, 24)
]

# ORT type string → numpy dtype (used when pre-allocating TRT output buffers)
_ORT_TYPE_TO_NP: Dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ONNX stitch inferencer with CUDA EP + IOBinding.")
    p.add_argument("--onnx-step", type=str, default="onnx_stitch/unetpp1.onnx", help="Path to step/update ONNX.")
    p.add_argument("--onnx-init", type=str, default=None,
                    help="Path to first_frame ONNX. Defaults to <onnx-step stem>_init.onnx")
    p.add_argument("--video", type=str, default="video01.mp4", help="Input video path.")
    p.add_argument("--start-frame", type=int, default=28000)
    p.add_argument("--end-frame", type=int, default=28100)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--fp16", action="store_true",
                    help="Load .fp16.onnx models exported by export_onnx.py.")
    p.add_argument("--out-video", type=str, default="trt_overlay.mp4")
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument("--debug", action="store_true", help="Render and save debug video.")
    p.add_argument("--trt", action="store_true",
                    help="Use TensorRT EP for both init and step models.  The step model "
                         "adds fusion-break outputs to prevent TRT NaN in the Gauss-Jordan "
                         "matrix inverse.")
    p.add_argument("--engine-step", type=str, default=None,
                    help="Pre-built TRT .engine for the step model (bypasses ONNX + TRT EP). "
                         "Example: trt_engine/unetpp.engine")
    p.add_argument("--engine-init", type=str, default=None,
                    help="Pre-built TRT .engine for the init model. "
                         "Defaults to <engine-step stem>_init.engine when --engine-step is set.")
    # Benchmark options (mutually exclusive with --debug)
    p.add_argument("--bench", action="store_true",
                    help="Run in benchmark mode: measure per-frame step latency and save JSON stats. "
                         "Mutually exclusive with --debug.")
    p.add_argument("--warmup", type=int, default=100,
                    help="Warmup frames before --start-frame (bench mode only, not included in stats).")
    p.add_argument("--bench-name", type=str, default="trt_bench.json",
                    help="Output JSON path for benchmark results (bench mode only).")
    return p.parse_args()


def _fix_uint8_for_cuda(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace UINT8 types with INT32 for broader EP compatibility."""
    changed = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and int(attr.i) == int(TensorProto.UINT8):
                    attr.i = int(TensorProto.INT32)
                    changed += 1
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in collection:
            if vi.type.HasField("tensor_type"):
                if vi.type.tensor_type.elem_type == TensorProto.UINT8:
                    vi.type.tensor_type.elem_type = TensorProto.INT32
                    changed += 1
    for init in model.graph.initializer:
        if init.data_type == TensorProto.UINT8:
            arr = numpy_helper.to_array(init).astype(np.int32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            changed += 1
    if changed:
        print(f"[fix] Patched {changed} UINT8 -> INT32 in {model.graph.name or 'model'}")
    return model


def _add_trt_fusion_break_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Expose Gauss-Jordan Div intermediates so TRT breaks its fusion.

    TensorRT fuses the entire DLT solver (including the 8x8 matrix inverse via
    Gauss-Jordan elimination) into a single kernel that computes NaN.  By adding
    the 8 pivot-division outputs, we force TRT to partition the graph at these
    points, which makes the computation correct.

    The added outputs are ignored at runtime -- they are only needed to influence
    TRT's graph partitioning.
    """
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        pass  # best-effort; value_info may already exist

    type_map = {vi.name: vi for vi in model.graph.value_info}
    existing = {o.name for o in model.graph.output}
    added = 0
    for name in _TRT_FUSION_BREAK_OUTPUTS:
        if name not in existing and name in type_map:
            model.graph.output.append(copy.deepcopy(type_map[name]))
            added += 1
    if added:
        print(f"[fix] Added {added} TRT fusion-break outputs (Gauss-Jordan Div intermediates)")
    return model


def _build_session(
    onnx_path: str, gpu_device_id: int, use_trt: bool = False,
    trt_fusion_break: bool = False,
) -> ort.InferenceSession:
    """Build an ORT session.

    When *use_trt* is True the TRT EP is tried first.
    When *trt_fusion_break* is True, Gauss-Jordan pivot outputs are added to
    prevent TRT from fusing the matrix-inverse kernel (which produces NaN).
    """
    model = onnx.load(onnx_path)
    model = _fix_uint8_for_cuda(model)
    if trt_fusion_break:
        model = _add_trt_fusion_break_outputs(model)
    model_bytes = model.SerializeToString()

    cuda_providers = [
        ("CUDAExecutionProvider", {"device_id": gpu_device_id}),
        "CPUExecutionProvider",
    ]

    if use_trt:
        trt_providers = [
            ("TensorrtExecutionProvider", {
                "device_id": gpu_device_id,
                "trt_fp16_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_max_workspace_size": str(2 << 30),
            }),
        ] + cuda_providers
        try:
            sess = ort.InferenceSession(model_bytes, providers=trt_providers)
            active = sess.get_providers()
            if "CUDAExecutionProvider" in active or "TensorrtExecutionProvider" in active:
                return sess
            print("[warn] TRT+CUDA session fell back to CPU, retrying with CUDA EP only")
        except Exception as e:
            print(f"[warn] TRT EP failed ({e}), falling back to CUDA EP")

    return ort.InferenceSession(model_bytes, providers=cuda_providers)


# ---------------------------------------------------------------------------
# TensorRT .engine direct-execution wrappers
# ---------------------------------------------------------------------------

class _NodeArgLike:
    """Minimal drop-in for ort.NodeArg exposing .name, .type, and .shape."""
    def __init__(self, name: str, ort_type: str, shape: List) -> None:
        self.name = name
        self.type = ort_type
        self.shape = shape


class _TRTIOBinding:
    """Minimal drop-in for ort.IOBinding used by TRTEngineSession.

    Inputs are stored as OrtValues (already on GPU).  The raw CUDA pointer
    (data_ptr()) is forwarded to the TRT execution context.  Outputs are
    pre-allocated on GPU as OrtValues; TRT writes directly into their memory.
    """

    def __init__(self, session: "TRTEngineSession") -> None:
        self._sess = session
        self._in: Dict[str, ort.OrtValue] = {}
        self._out: Dict[str, ort.OrtValue] = {}

    def bind_ortvalue_input(self, name: str, ortvalue: ort.OrtValue) -> None:
        self._in[name] = ortvalue

    def bind_output(self, name: str, device: str, device_id: int) -> None:
        meta = self._sess._out_meta[name]
        dtype = _ORT_TYPE_TO_NP.get(meta.type, np.float32)
        shape = tuple(int(d) for d in meta.shape)
        buf = np.zeros(shape, dtype=dtype)
        self._out[name] = ort.OrtValue.ortvalue_from_numpy(buf, device, device_id)

    def synchronize_inputs(self) -> None:
        pass  # OrtValues are already resident on GPU

    def synchronize_outputs(self) -> None:
        pass  # execute_async_v3 + torch.cuda.synchronize covers this

    def get_outputs(self) -> List[ort.OrtValue]:
        return [self._out[n] for n in self._sess._out_names]


class TRTEngineSession:
    """Serialized TRT .engine wrapped with an ort.InferenceSession-compatible
    interface (get_inputs / get_outputs / io_binding / run_with_iobinding).

    Requires TensorRT 10.x (num_io_tensors / get_tensor_* / execute_async_v3).
    """

    def __init__(self, engine_path: str, gpu_id: int) -> None:
        import tensorrt as trt  # local import – not required at module level

        self._gpu_id = gpu_id
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=gpu_id)

        # Build dtype map from the live TRT module
        trt_dtype_to_ort: Dict = {
            trt.DataType.FLOAT: "tensor(float)",
            trt.DataType.HALF:  "tensor(float16)",
            trt.DataType.INT8:  "tensor(int8)",
            trt.DataType.INT32: "tensor(int32)",
            trt.DataType.INT64: "tensor(int64)",
            trt.DataType.BOOL:  "tensor(bool)",
        }

        self._in_names: List[str] = []
        self._out_names: List[str] = []
        self._in_meta: Dict[str, _NodeArgLike] = {}
        self._out_meta: Dict[str, _NodeArgLike] = {}

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = list(self._engine.get_tensor_shape(name))
            ort_type = trt_dtype_to_ort.get(self._engine.get_tensor_dtype(name), "tensor(float)")
            meta = _NodeArgLike(name, ort_type, shape)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_names.append(name)
                self._in_meta[name] = meta
            else:
                self._out_names.append(name)
                self._out_meta[name] = meta

    def get_inputs(self) -> List[_NodeArgLike]:
        return [self._in_meta[n] for n in self._in_names]

    def get_outputs(self) -> List[_NodeArgLike]:
        return [self._out_meta[n] for n in self._out_names]

    def get_providers(self) -> List[str]:
        return ["TensorrtExecutionProvider"]

    def io_binding(self) -> _TRTIOBinding:
        return _TRTIOBinding(self)

    def run_with_iobinding(self, iob: _TRTIOBinding) -> None:
        for name, ortval in iob._in.items():
            self._context.set_tensor_address(name, ortval.data_ptr())
        for name, ortval in iob._out.items():
            self._context.set_tensor_address(name, ortval.data_ptr())
        self._context.execute_async_v3(self._stream.cuda_stream)
        torch.cuda.synchronize(self._gpu_id)


def _build_trt_session(engine_path: str, gpu_id: int) -> TRTEngineSession:
    """Load a serialized TensorRT .engine and wrap it in a session object."""
    print(f"[info] Loading TRT engine: {engine_path}")
    sess = TRTEngineSession(engine_path, gpu_id)
    print(f"[info]   inputs : {[m.name for m in sess.get_inputs()]}")
    print(f"[info]   outputs: {[m.name for m in sess.get_outputs()]}")
    return sess


# ---------------------------------------------------------------------------
# IOBinding helpers (state stays on GPU as OrtValue between iterations)
# ---------------------------------------------------------------------------

def _ortvalue_from_numpy(arr: np.ndarray, ort_type: str, use_fp16: bool, gpu_id: int) -> ort.OrtValue:
    casted = _cast_for_input(arr, ort_type, use_fp16)
    return ort.OrtValue.ortvalue_from_numpy(casted, "cuda", gpu_id)


def _as_numpy(value: Union[np.ndarray, ort.OrtValue]) -> np.ndarray:
    if isinstance(value, ort.OrtValue):
        return value.numpy()
    return value


def _run_init_iobinding(
    init_sess: ort.InferenceSession,
    frame_np: np.ndarray,
    ellipse_np: np.ndarray,
    use_fp16: bool,
    gpu_id: int,
) -> Tuple[Dict[str, ort.OrtValue], Optional[ort.OrtValue]]:
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
    state = {
        "canvas": out["canvas"],
        "canvas_mask": out["canvas_mask"],
        "H_cum_curr": out["H_cum_curr"],
        "prev_keypoints": out["prev_keypoints"],
        "prev_descriptors": out["prev_descriptors"],
    }
    return state, out.get("seg_map")


def _run_step_iobinding(
    step_sess: ort.InferenceSession,
    inmeta: Dict[str, ort.NodeArg],
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

    # Derive engine-init path from engine-step when only step is given
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
        # ONNX path: optionally use TRT EP (--trt) with fusion-break workaround
        print(f"[info] step={args.onnx_step}, init={args.onnx_init}")
        step_sess = _build_session(
            args.onnx_step, args.gpu_device_id,
            use_trt=args.trt, trt_fusion_break=args.trt,
        )
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
        desc = "bench-engine" if args.engine_step else ("bench-TRT" if args.trt else "bench-CUDA")
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
                state, _ = _run_init_iobinding(
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
                state, _ = _run_init_iobinding(
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
    desc = "engine" if args.engine_step else ("TRT" if args.trt else "CUDA")
    pbar = tqdm(total=total, desc=desc)

    frame_idx = args.start_frame
    while frame_idx <= args.end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_np, ellipse_np, roi, frame_roi_bgr = _preprocess_frame(frame_bgr, roi)
        frame_np, ellipse_np = _resize_to_expected(frame_np, ellipse_np, exp_h, exp_w)

        if state is None:
            state, _init_seg = _run_init_iobinding(
                init_sess, frame_np, ellipse_np, args.fp16, args.gpu_device_id,
            )

        out = _run_step_iobinding(
            step_sess, inmeta, out_names,
            frame_np, ellipse_np, state,
            args.fp16, args.gpu_device_id,
        )

        # Check needs_reset flag
        needs_reset_ort = out.get("needs_reset")
        needs_reset = False
        if needs_reset_ort is not None:
            nr = _as_numpy(needs_reset_ort)
            needs_reset = bool(nr.item()) if nr.size == 1 else bool(nr.any())

        if needs_reset:
            state, init_seg_ort = _run_init_iobinding(
                init_sess, frame_np, ellipse_np, args.fp16, args.gpu_device_id,
            )
            seg_map_ort = init_seg_ort
        else:
            _update_state_from_outputs(state, out)
            seg_map_ort = out.get("seg_map")

        if args.debug:
            seg_map = _as_numpy(seg_map_ort) if seg_map_ort is not None else None
            if seg_map is None:
                raise RuntimeError("seg_map was not found in outputs.")
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
            oh = overlay.shape[0]
            if canvas_bgr.shape[0] != oh:
                new_w = max(1, int(round(canvas_bgr.shape[1] * oh / canvas_bgr.shape[0])))
                canvas_bgr = cv2.resize(canvas_bgr, (new_w, oh), interpolation=cv2.INTER_LINEAR)
            vis = cv2.hconcat([overlay, canvas_bgr])

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
