#!/usr/bin/env python3
"""Export ONNX models to TensorRT .engine files.

The step model contains homography estimation (DLT solver via Gauss-Jordan
matrix inverse).  TensorRT's kernel fusion incorrectly handles the Gauss-Jordan
elimination Div chain, producing NaN.  The workaround is to expose the 8
pivot-division intermediate tensors (Div_16..Div_23) as model outputs, which
forces TRT to break its fusion around the matrix inverse and compute correctly.
"""
import argparse
import copy
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

try:
    import tensorrt as trt
except ImportError:
    print("tensorrt package not found. Install with: pip install tensorrt")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TensorRT engines from ONNX models.")
    p.add_argument("--onnx-step", type=str, default="onnx_stitch/unetpp.onnx",
                    help="Path to step/update ONNX model.")
    p.add_argument("--onnx-init", type=str, default=None,
                    help="Path to first_frame ONNX. Defaults to <onnx-step stem>_init.onnx")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 precision.")
    p.add_argument("--workspace", type=int, default=4, help="Workspace size in GB.")
    p.add_argument("--out-dir", type=str, default="trt_engine", help="Output directory for .engine files.")
    return p.parse_args()


def fix_uint8_for_trt(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace UINT8 intermediate types with INT32 for TRT compatibility.

    TensorRT does not support UINT8 as a general intermediate tensor type.
    This patches Cast nodes, outputs, value_info, and initializers.

    Graph inputs that are UINT8 are NOT converted directly to INT32 because
    TRT would misclassify large INT32 inputs (e.g. canvas_mask) as "shape
    tensors".  Instead, the input type is changed to FLOAT32 (which TRT
    cannot classify as a shape tensor), with a Cast(FLOAT32→INT32) node
    inserted right after it.
    """
    from onnx import helper
    changed = 0

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and int(attr.i) == int(TensorProto.UINT8):
                    attr.i = int(TensorProto.INT32)
                    changed += 1

    for collection in (model.graph.output, model.graph.value_info):
        for vi in collection:
            if vi.type.HasField("tensor_type"):
                if vi.type.tensor_type.elem_type == TensorProto.UINT8:
                    vi.type.tensor_type.elem_type = TensorProto.INT32
                    changed += 1

    cast_nodes = []
    for inp in model.graph.input:
        if not inp.type.HasField("tensor_type"):
            continue
        if inp.type.tensor_type.elem_type != TensorProto.UINT8:
            continue
        inp.type.tensor_type.elem_type = int(TensorProto.FLOAT)
        cast_out_name = inp.name + "_int32"
        cast_node = helper.make_node(
            "Cast",
            inputs=[inp.name],
            outputs=[cast_out_name],
            name=f"_float_to_int32_{inp.name}",
            to=int(TensorProto.INT32),
        )
        cast_nodes.append(cast_node)
        for node in model.graph.node:
            for i, node_inp in enumerate(node.input):
                if node_inp == inp.name:
                    node.input[i] = cast_out_name
        changed += 1

    if cast_nodes:
        existing_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(cast_nodes)
        model.graph.node.extend(existing_nodes)

    for init in model.graph.initializer:
        if init.data_type == TensorProto.UINT8:
            arr = numpy_helper.to_array(init).astype(np.int32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            changed += 1

    if changed:
        print(f"[fix] Replaced {changed} UINT8 types with INT32 for TRT compatibility")

    return model


# Gauss-Jordan pivot-division outputs whose exposure forces TRT to break its
# kernel fusion around the 8x8 matrix inverse in the DLT homography solver.
_TRT_FUSION_BREAK_OUTPUTS: List[str] = [
    f"/Div_{i}_output_0" for i in range(16, 24)
]


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


def build_engine(onnx_path: str, engine_path: str, fp16: bool = False, workspace_gb: int = 4,
                 fusion_break: bool = False):
    """Build a TensorRT engine from an ONNX model."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"[build] Parsing {onnx_path} ...")
    model = onnx.load(onnx_path)
    model = fix_uint8_for_trt(model)
    if fusion_break:
        model = _add_trt_fusion_break_outputs(model)
    serialized_onnx = model.SerializeToString()

    if not parser.parse(serialized_onnx):
        for i in range(parser.num_errors):
            print(f"  Parse error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    except AttributeError:
        config.max_workspace_size = workspace_gb << 30

    if fp16:
        if builder.platform_has_fast_fp16:
            print("[build] FP16 enabled (hardware fast path available)")
        else:
            print("[build] FP16 enabled (no hardware fast path, may be slow)")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"[build] Building engine (fp16={fp16}, workspace={workspace_gb}GB) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed: {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = Path(engine_path).stat().st_size / 1e6
    print(f"[done] {engine_path} ({size_mb:.1f} MB)")


def main():
    args = parse_args()

    if args.onnx_init is None:
        p = Path(args.onnx_step)
        args.onnx_init = str(p.with_name(f"{p.stem}_init{p.suffix}"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_engine = out_dir / Path(args.onnx_step).with_suffix(".engine").name
    init_engine = out_dir / Path(args.onnx_init).with_suffix(".engine").name

    print(f"[info] step: {args.onnx_step} -> {step_engine}")
    print(f"[info] init: {args.onnx_init} -> {init_engine}")
    print(f"[info] TensorRT {trt.__version__}")

    build_engine(args.onnx_step, str(step_engine), args.fp16, args.workspace,
                 fusion_break=True)
    build_engine(args.onnx_init, str(init_engine), args.fp16, args.workspace)


if __name__ == "__main__":
    main()
