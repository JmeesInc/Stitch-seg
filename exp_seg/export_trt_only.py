#!/usr/bin/env python3
"""Export Stitcher_ONNX (only) ONNX models to TensorRT .engine files."""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

try:
    import tensorrt as trt
except ImportError:
    print("tensorrt package not found. Install with: pip install tensorrt")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TensorRT engines from Stitcher_ONNX (only) models.")
    p.add_argument("--onnx-step", type=str, default="onnx_only/only.onnx",
                    help="Path to step/update ONNX model.")
    p.add_argument("--onnx-init", type=str, default=None,
                    help="Path to first_frame ONNX. Defaults to <onnx-step stem>_init.onnx")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 precision.")
    p.add_argument("--workspace", type=int, default=16, help="Workspace size in GB.")
    p.add_argument("--out-dir", type=str, default="trt_engine_only", help="Output directory for .engine files.")
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

    # 1. Patch Cast nodes that produce UINT8
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and int(attr.i) == int(TensorProto.UINT8):
                    attr.i = int(TensorProto.INT32)
                    changed += 1

    # 2. Patch graph outputs and value_info (but NOT graph inputs)
    for collection in (model.graph.output, model.graph.value_info):
        for vi in collection:
            if vi.type.HasField("tensor_type"):
                if vi.type.tensor_type.elem_type == TensorProto.UINT8:
                    vi.type.tensor_type.elem_type = TensorProto.INT32
                    changed += 1

    # 3. For UINT8 graph inputs, insert Cast(UINT8→INT32) node and redirect
    #    consumers to use the casted tensor.
    # For UINT8 graph inputs: change to FLOAT32 (not INT32, which TRT
    # misclassifies as a shape tensor), then Cast FLOAT32→INT32 internally.
    cast_nodes = []
    for inp in model.graph.input:
        if not inp.type.HasField("tensor_type"):
            continue
        if inp.type.tensor_type.elem_type != TensorProto.UINT8:
            continue
        # Change input type to FLOAT32 (avoids shape tensor classification)
        inp.type.tensor_type.elem_type = int(TensorProto.FLOAT)
        # Insert Cast(FLOAT32→INT32) right after input
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

    # Insert cast nodes at the beginning of the graph
    if cast_nodes:
        existing_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(cast_nodes)
        model.graph.node.extend(existing_nodes)

    # 4. Patch initializers
    for init in model.graph.initializer:
        if init.data_type == TensorProto.UINT8:
            arr = numpy_helper.to_array(init).astype(np.int32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            changed += 1

    if changed:
        print(f"[fix] Replaced {changed} UINT8 types with INT32 for TRT compatibility")

    return model


def build_engine(onnx_path: str, engine_path: str, fp16: bool = False, workspace_gb: int = 4):
    """Build a TensorRT engine from an ONNX model.

    Automatically creates an optimization profile when the network has
    inputs with dynamic dimensions (required by TensorRT).
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"[build] Parsing {onnx_path} ...")
    model = onnx.load(onnx_path)
    model = fix_uint8_for_trt(model)
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

    # --- Optimization profile for dynamic / shape inputs ---
    # TensorRT requires at least one profile when any input (or internal
    # shape tensor) has dynamic dimensions.  We inspect every network input
    # and set min=opt=max to the static shape (replacing -1 with a sensible
    # default).  This also covers the "shape inputs" that TRT infers from
    # ONNX Shape/ConstantOfShape/Expand ops.
    # Always add an optimization profile — it is harmless for fully-static
    # networks and required when TRT detects dynamic dims or shape inputs.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = inp.shape
        # Replace any -1 dims with a concrete default (use ONNX metadata).
        static_shape = tuple(d if d != -1 else 1 for d in shape)
        if inp.is_shape_tensor:
            # Shape tensors need set_shape_input (values, not dimensions).
            vals = [d if d != -1 else 1 for d in shape]
            profile.set_shape_input(name, vals, vals, vals)
            print(f"  [profile] shape-input {name}: {vals}")
        else:
            profile.set_shape(name, static_shape, static_shape, static_shape)
            print(f"  [profile] {name}: {list(static_shape)}")
    config.add_optimization_profile(profile)

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

    build_engine(args.onnx_step, str(step_engine), args.fp16, args.workspace)
    build_engine(args.onnx_init, str(init_engine), args.fp16, args.workspace)


if __name__ == "__main__":
    main()
