#!/usr/bin/env python3
"""ONNX export utilities for segmentation model / stitch inferencer."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from onnxsim import simplify


def _save_onnx(model: onnx.ModelProto, path: str | Path) -> None:
    """Save ONNX model, using a single external data file if >2GB protobuf limit."""
    path = Path(path)
    try:
        onnx.save(model, str(path))
    except ValueError:
        # Model too large for protobuf — save with external data in one file
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}.data",
        )
        print(f"  (saved with external data: {path.name}.data)")


def _cleanup_external_data(directory: Path) -> None:
    """Remove stray external data files created by torch.onnx.export / onnx.save."""
    import glob
    patterns = ["*_attr__value", "*_attr__value*", "*Constant*attr*"]
    removed = 0
    for pat in patterns:
        for f in glob.glob(str(directory / pat)):
            Path(f).unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"_cleanup_external_data: removed {removed} stray files")


def _resave_without_external_data(path: Path) -> None:
    """Reload an ONNX model and re-save it as a single self-contained file.

    torch.onnx.export may write large initializers as external data files.
    This reloads the model (which reads external data back in) and saves
    it as a single .onnx file, then cleans up any leftover external files.
    """
    onnx_model = onnx.load(str(path), load_external_data=True)
    # Convert any external data references to internal
    from onnx.external_data_helper import convert_model_from_external_data
    convert_model_from_external_data(onnx_model)
    _save_onnx(onnx_model, path)
    _cleanup_external_data(path.parent)


from stitch_seg.onnx_exporters import register_deform_conv2d_onnx_op, register_cumprod_onnx_op
register_deform_conv2d_onnx_op()
register_cumprod_onnx_op()



class FirstFrameExportWrapper(nn.Module):
    """Wrapper to export inferencer.first_frame as a standalone ONNX graph."""

    def __init__(self, inferencer):
        super().__init__()
        self.inferencer = inferencer

    def forward(self, frame: torch.Tensor, ellipse_mask: torch.Tensor):
        canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors = self.inferencer.first_frame(frame, ellipse_mask)
        return canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors



def build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.device = torch.device(args.device)
    cfg.num_classes = args.num_classes
    cfg.backbone = args.backbone
    cfg.segmentation_weights = args.seg_weights
    cfg.tool_detector_weights = args.tool_weights
    cfg.port_detector_weights = args.port_weights
    cfg.apply_ellipse_mask = True
    cfg.tool_class_ch = 0
    cfg.laplacian_var_min = args.laplacian_var_min
    cfg.canvas_superres_scale = 1.0
    cfg.canvas_scale_x = 3.0
    cfg.canvas_scale_y = 3.0
    cfg.gradient_radius = 201
    cfg.canvas_border_trim_px = 12
    cfg.reset_shear_angle = 15.0
    cfg.reset_rotate_angle = 15.0
    cfg.reset_scale_factor = 2.0
    return cfg


def check_export(path: Path):
    if onnx is None:
        print("onnx is not installed; skip checker.")
        return
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    print(f"onnx checker passed: {path}")


def force_slice_indices_int64(path: Path):
    model = onnx.load(str(path))
    graph = model.graph
    rewritten_nodes = []
    changed = 0
    output_to_node = {out: node for node in graph.node for out in node.output}
    initializer_dtype = {init.name: init.data_type for init in graph.initializer}

    def _constant_dtype(node):
        for attr in node.attribute:
            if attr.name == "value":
                return attr.t.data_type
        return None

    passthrough_ops = {"Identity", "Unsqueeze", "Squeeze", "Reshape"}

    def _is_known_int64_tensor(name: str, depth: int = 4) -> bool:
        if depth <= 0:
            return False
        if not name:
            return False
        dtype = initializer_dtype.get(name)
        if dtype is not None:
            return dtype == TensorProto.INT64
        producer = output_to_node.get(name)
        if producer is None:
            return False
        if producer.op_type == "Cast":
            for attr in producer.attribute:
                if attr.name == "to":
                    return int(attr.i) == int(TensorProto.INT64)
            return False
        if producer.op_type == "Constant":
            return _constant_dtype(producer) == TensorProto.INT64
        if producer.op_type in passthrough_ops and len(producer.input) > 0:
            return _is_known_int64_tensor(producer.input[0], depth=depth - 1)
        return False

    for node in graph.node:
        if node.op_type != "Slice":
            rewritten_nodes.append(node)
            continue

        new_inputs = list(node.input)
        cast_nodes = []
        node_name = node.name if node.name else f"Slice_{changed}"

        for inp_idx in (1, 2, 3, 4):  # starts, ends, axes, steps
            if inp_idx >= len(new_inputs):
                continue
            src = new_inputs[inp_idx]
            if not src or _is_known_int64_tensor(src):
                continue
            cast_out = f"{src}__int64_for_{node_name}_{inp_idx}"
            cast_name = f"{node_name}_CastInt64_{inp_idx}"
            cast_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[src],
                    outputs=[cast_out],
                    name=cast_name,
                    to=TensorProto.INT64,
                )
            )
            new_inputs[inp_idx] = cast_out

        if cast_nodes:
            changed += 1
            rewritten_nodes.extend(cast_nodes)
            patched = onnx.NodeProto()
            patched.CopyFrom(node)
            patched.input[:] = new_inputs
            rewritten_nodes.append(patched)
        else:
            rewritten_nodes.append(node)

    if changed > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)
        _save_onnx(model, path)
        print(f"patched Slice index dtype to int64: {path} (nodes={changed})")


def fix_leaked_parameters(
    path: Path,
    model: nn.Module,
    expected_input_names: list[str],
) -> int:
    """Fix parameters that leaked into ONNX graph.input instead of initializer.

    PyTorch 2.9+'s TorchScript-based ONNX tracer sometimes fails to embed
    model parameters as initializers for complex models, causing them to
    appear as graph inputs.  This post-processes the ONNX file:

    1. Any graph.input that is already an initializer but not an expected
       input is removed from graph.input (legacy ONNX convention cleanup).
    2. Any remaining graph.input not in *expected_input_names* is looked up
       in the PyTorch model's state_dict and embedded as an initializer.
    """
    import numpy as np
    from onnx import numpy_helper

    onnx_model = onnx.load(str(path))
    graph = onnx_model.graph

    expected = set(expected_input_names)
    existing_inits = {init.name for init in graph.initializer}

    # Build state_dict lookup (full name + suffix matching)
    sd = {}
    for name, param in model.state_dict().items():
        sd[name] = param.detach().cpu().numpy()
    for name, param in model.named_parameters():
        if name not in sd:
            sd[name] = param.detach().cpu().numpy()
    for name, buf in model.named_buffers():
        if name not in sd:
            sd[name] = buf.detach().cpu().numpy()

    suffix_map: dict[str, np.ndarray] = {}
    for name, arr in sd.items():
        parts = name.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            if suffix not in suffix_map:
                suffix_map[suffix] = arr

    removed_existing = 0
    embedded_new = 0
    not_found = []

    for inp in list(graph.input):
        if inp.name in expected:
            continue
        # Already an initializer — just needs removal from graph.input
        if inp.name in existing_inits:
            removed_existing += 1
            continue
        # Not an initializer — try to embed from state_dict
        arr = sd.get(inp.name) or suffix_map.get(inp.name)
        if arr is None:
            inp_parts = inp.name.split(".")
            for i in range(len(inp_parts)):
                suffix = ".".join(inp_parts[i:])
                if suffix in suffix_map:
                    arr = suffix_map[suffix]
                    break
        if arr is None:
            not_found.append(inp.name)
            continue
        tensor = numpy_helper.from_array(arr, name=inp.name)
        graph.initializer.append(tensor)
        embedded_new += 1

    # Remove all non-expected inputs from graph.input
    total_removed = removed_existing + embedded_new
    if total_removed > 0:
        inputs_to_keep = [inp for inp in graph.input if inp.name in expected]
        del graph.input[:]
        graph.input.extend(inputs_to_keep)
        _save_onnx(onnx_model, path)
        print(f"fix_leaked_parameters ({path.name}): "
              f"removed {removed_existing} existing initializers from graph.input, "
              f"embedded {embedded_new} new parameters as initializers")

    if not_found:
        print(f"  WARNING: {len(not_found)} leaked inputs not found in state_dict:")
        for name in not_found[:10]:
            print(f"    {name}")
        if len(not_found) > 10:
            print(f"    ... and {len(not_found) - 10} more")

    return total_removed


def ort_smoke_test(path: Path, input_names: list[str], input_tensors: tuple[torch.Tensor, ...]):
    if ort is None:
        print("onnxruntime is not installed; skip ORT smoke test.")
        return
    providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    provided = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, input_tensors)}
    required_names = [x.name for x in sess.get_inputs()]
    ort_inputs = {name: provided[name] for name in required_names if name in provided}
    _ = sess.run(None, ort_inputs)
    print(f"onnxruntime smoke test passed: {path}")


def strip_non_input_graph_inputs(path: Path, expected_input_names: list[str]) -> int:
    """Remove any graph.input entries that are not real model inputs.

    After simplification or FP16 conversion, some initializers may reappear
    in graph.input.  This strips them unconditionally.
    """
    onnx_model = onnx.load(str(path))
    graph = onnx_model.graph
    expected = set(expected_input_names)
    leaked = [inp for inp in graph.input if inp.name not in expected]
    if leaked:
        inputs_to_keep = [inp for inp in graph.input if inp.name in expected]
        del graph.input[:]
        graph.input.extend(inputs_to_keep)
        _save_onnx(onnx_model, path)
        print(f"strip_non_input_graph_inputs ({path.name}): removed {len(leaked)} entries")
    return len(leaked)


def _ensure_node_names(model: onnx.ModelProto) -> None:
    """Assign unique names to unnamed nodes so node_block_list can reference them."""
    idx = 0
    for node in model.graph.node:
        if not node.name:
            node.name = f"_unnamed_{idx}"
            idx += 1


def _trace_fp32_nodes(model: onnx.ModelProto, src: str, dst: str) -> list[str]:
    """Find nodes that should remain in FP32 for the homography subgraph.

    Uses backward reachability from *dst* (H_cum_next), excluding nodes that
    belong to known FP16-safe subgraphs (extractor, matcher, masking/seg models).
    These subgraphs produce feature points and scores that are fine in FP16;
    only the homography estimation (matrix inversion, DLT solve) and
    accumulation need FP32 precision.
    """
    from collections import deque

    # Prefixes for subgraphs that are safe in FP16
    _FP16_SAFE_PREFIXES = (
        "/extractor/", "/matcher/",
        "/masking_model/", "/masking_model2/",
        "/model/",  # seg model
    )

    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Backward from dst, skipping FP16-safe subgraph nodes
    bwd_nodes: set[str] = set()
    q: deque[str] = deque([dst])
    visited: set[str] = set()
    while q:
        t = q.popleft()
        if t in visited:
            continue
        visited.add(t)
        if t in output_to_node:
            node = output_to_node[t]
            if node.name in bwd_nodes:
                continue
            # Skip nodes in FP16-safe subgraphs
            if node.name.startswith(_FP16_SAFE_PREFIXES):
                continue
            bwd_nodes.add(node.name)
            for inp in node.input:
                q.append(inp)

    return list(bwd_nodes)


def _patch_subgraph_to_fp32(
    model: onnx.ModelProto,
    node_names: set[str],
) -> None:
    """Convert a subgraph of nodes from FP16 back to FP32 after global FP16 conversion.

    For every node whose name is in *node_names*:
      - Change its FP16 initializer inputs to FP32 (duplicating if shared).
      - Update value_info for its outputs to FP32.
    Then insert Cast(FP16→FP32) at inputs coming from outside the subgraph
    and Cast(FP32→FP16) at outputs going to nodes outside the subgraph.
    """
    import numpy as np
    from onnx import numpy_helper

    node_by_name = {n.name: n for n in model.graph.node}
    patched_nodes = {n.name: n for n in model.graph.node if n.name in node_names}

    # Build tensor → producer/consumer maps
    tensor_producer: dict[str, str] = {}  # tensor_name → node_name
    tensor_consumers: dict[str, list[str]] = {}  # tensor_name → [node_names]
    for node in model.graph.node:
        for out in node.output:
            tensor_producer[out] = node.name
        for inp in node.input:
            tensor_consumers.setdefault(inp, []).append(node.name)

    # Model inputs (graph-level)
    graph_input_names = {i.name for i in model.graph.input}

    # Initializer map
    init_map = {i.name: i for i in model.graph.initializer}

    # value_info map
    vi_map = {}
    for v in model.graph.value_info:
        vi_map[v.name] = v

    # 1. Collect all tensors produced by patched nodes (make them FP32)
    patched_outputs: set[str] = set()
    for node in patched_nodes.values():
        for out in node.output:
            patched_outputs.add(out)

    # 2. Collect all tensors consumed by patched nodes
    patched_inputs: set[str] = set()
    for node in patched_nodes.values():
        for inp in node.input:
            patched_inputs.add(inp)

    # 3. Update value_info: patched node outputs → FP32 (only float16 tensors)
    fp32_tensors: set[str] = set()
    graph_output_names = {o.name for o in model.graph.output}
    graph_output_type = {}
    for o in model.graph.output:
        if o.type.HasField("tensor_type"):
            graph_output_type[o.name] = o.type.tensor_type.elem_type
    for tname in patched_outputs:
        is_fp16 = False
        if tname in vi_map:
            vi = vi_map[tname]
            if vi.type.tensor_type.elem_type == int(TensorProto.FLOAT16):
                vi.type.tensor_type.elem_type = int(TensorProto.FLOAT)
                is_fp16 = True
        elif graph_output_type.get(tname) == int(TensorProto.FLOAT16):
            is_fp16 = True
        if is_fp16:
            fp32_tensors.add(tname)

    # 4. Handle initializers consumed by patched nodes: convert FP16 → FP32
    for tname in patched_inputs:
        if tname not in init_map:
            continue
        init_tensor = init_map[tname]
        if init_tensor.data_type != int(TensorProto.FLOAT16):
            continue
        consumers = tensor_consumers.get(tname, [])
        non_patched_consumers = [c for c in consumers if c not in node_names]
        if non_patched_consumers:
            arr = numpy_helper.to_array(init_tensor).astype(np.float32)
            new_name = tname + "_fp32"
            new_init = numpy_helper.from_array(arr, name=new_name)
            model.graph.initializer.append(new_init)
            for node in patched_nodes.values():
                for i, inp in enumerate(node.input):
                    if inp == tname:
                        node.input[i] = new_name
        else:
            arr = numpy_helper.to_array(init_tensor).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=tname)
            init_tensor.CopyFrom(new_init)

    # 5. Insert Cast nodes at boundaries
    new_nodes = []
    cast_cache_in: dict[str, str] = {}
    cast_cache_out: dict[str, str] = {}

    for node in list(patched_nodes.values()):
        for i, inp in enumerate(node.input):
            if inp in init_map:
                continue
            if inp in graph_input_names:
                producer = None
            else:
                producer = tensor_producer.get(inp)
            if producer is not None and producer in node_names:
                continue
            if inp in cast_cache_in:
                node.input[i] = cast_cache_in[inp]
                continue
            inp_type = None
            if inp in vi_map:
                inp_type = vi_map[inp].type.tensor_type.elem_type
            elif inp in graph_input_names:
                for gi in model.graph.input:
                    if gi.name == inp:
                        inp_type = gi.type.tensor_type.elem_type
                        break
            if inp_type != int(TensorProto.FLOAT16):
                continue
            cast_out = inp + "_cast_fp32"
            cast_node = onnx.helper.make_node(
                "Cast", inputs=[inp], outputs=[cast_out],
                to=int(TensorProto.FLOAT), name=f"_hcum_cast_in_{len(new_nodes)}"
            )
            new_nodes.append(cast_node)
            cast_cache_in[inp] = cast_out
            node.input[i] = cast_out

    for node in list(patched_nodes.values()):
        for j, out in enumerate(node.output):
            if out not in fp32_tensors:
                continue
            consumers = tensor_consumers.get(out, [])
            non_patched = [c for c in consumers if c not in node_names]
            is_graph_output = out in graph_output_names
            if not non_patched and not is_graph_output:
                continue

            if out in cast_cache_out:
                fp16_name = cast_cache_out[out]
            else:
                fp16_name = out + "_cast_fp16"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[out], outputs=[fp16_name],
                    to=int(TensorProto.FLOAT16), name=f"_hcum_cast_out_{len(new_nodes)}"
                )
                new_nodes.append(cast_node)
                if out in vi_map:
                    vi_orig = vi_map[out]
                    vi_new = onnx.helper.make_tensor_value_info(
                        fp16_name, TensorProto.FLOAT16,
                        [d.dim_value if d.dim_value else d.dim_param
                         for d in vi_orig.type.tensor_type.shape.dim] if vi_orig.type.tensor_type.HasField("shape") else None
                    )
                    model.graph.value_info.append(vi_new)
                cast_cache_out[out] = fp16_name

            for cname in non_patched:
                cnode = node_by_name[cname]
                for k, ci in enumerate(cnode.input):
                    if ci == out:
                        cnode.input[k] = fp16_name

            if is_graph_output:
                internal_name = out + "_fp32_internal"
                for k2, o2 in enumerate(node.output):
                    if o2 == out:
                        node.output[k2] = internal_name
                        break
                if out in vi_map:
                    vi_map[out].name = internal_name
                new_nodes = [n for n in new_nodes if fp16_name not in n.output]
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[internal_name], outputs=[out],
                    to=int(TensorProto.FLOAT16),
                    name=f"_hcum_cast_out_graphout_{out}"
                )
                new_nodes.append(cast_node)
                for cname in non_patched:
                    cnode = node_by_name[cname]
                    for k, ci in enumerate(cnode.input):
                        if ci == fp16_name:
                            cnode.input[k] = out
                for pnode in patched_nodes.values():
                    for k, pi in enumerate(pnode.input):
                        if pi == out:
                            pnode.input[k] = internal_name

    model.graph.node.extend(new_nodes)

    # Re-sort nodes topologically
    from collections import deque
    all_nodes = list(model.graph.node)
    available = set(graph_input_names) | set(init_map.keys())
    for n in all_nodes:
        for inp in n.input:
            if inp.endswith("_fp32") and inp not in available:
                for init in model.graph.initializer:
                    if init.name == inp:
                        available.add(inp)
    sorted_nodes = []
    remaining = list(all_nodes)
    max_iters = len(remaining) + 1
    for _ in range(max_iters):
        if not remaining:
            break
        next_remaining = []
        for node in remaining:
            if all(inp in available or inp == "" for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    available.add(out)
            else:
                next_remaining.append(node)
        if len(next_remaining) == len(remaining):
            sorted_nodes.extend(next_remaining)
            break
        remaining = next_remaining

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)

    print(f"_patch_subgraph_to_fp32: patched {len(patched_nodes)} nodes, "
          f"inserted {len(new_nodes)} Cast nodes")


def fix_fp16_cast_to_mismatch(model: onnx.ModelProto, skip_prefixes: tuple[str, ...] = ()) -> int:
    """Fix known onnxconverter-common FP16 conversion issue.

    `float16.convert_float_to_float16()` sometimes updates `value_info` to FP16 but
    leaves `Cast(to=FLOAT)` nodes unchanged, causing ORT type validation failures.
    We patch those Cast nodes to `to=FLOAT16` when their outputs are annotated FP16.

    Nodes whose names start with any of *skip_prefixes* are left untouched
    (used to preserve intentional FP32 boundary casts).
    """
    vi_elem_type = {}
    for v in model.graph.value_info:
        if not v.name or not v.type.HasField("tensor_type"):
            continue
        vi_elem_type[v.name] = v.type.tensor_type.elem_type

    patched = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = None
        for attr in node.attribute:
            if attr.name == "to":
                to_attr = attr
                break
        if to_attr is None:
            continue
        if int(to_attr.i) == int(TensorProto.FLOAT16):
            continue
        if skip_prefixes and node.name.startswith(skip_prefixes):
            continue
        if any(vi_elem_type.get(out) == TensorProto.FLOAT16 for out in node.output):
            to_attr.i = int(TensorProto.FLOAT16)
            patched += 1
    return patched


def export_stitch_inferencer(args: argparse.Namespace):
    from stitch_seg import Stitcher_ONNX

    cfg = build_cfg(args)
    inferencer = Stitcher_ONNX(cfg=cfg, input_size=(args.height, args.width)).eval()

    h, w = args.height, args.width
    frame = torch.randint(0, 256, (1, 3, h, w), device=cfg.device, dtype=torch.float32)
    ellipse_mask = torch.zeros((1, 1, h, w), device=cfg.device, dtype=torch.float32)
    with torch.no_grad():
        canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors = (
            inferencer.first_frame(frame, ellipse_mask)
        )

    # Export with at least 1 keypoint to avoid degenerate tracing shape.
    if prev_keypoints.shape[1] == 0:
        desc_dim = prev_descriptors.shape[-1] if prev_descriptors.ndim == 3 else 128
        prev_keypoints = torch.zeros((1, 1, 2), device=cfg.device, dtype=torch.float32)
        prev_descriptors = torch.zeros((1, 1, desc_dim), device=cfg.device, dtype=torch.float32)

    step_inputs = (
        frame,
        ellipse_mask,
        H_cum_curr.to(torch.float32),
        prev_keypoints.to(torch.float32),
        prev_descriptors.to(torch.float32),
        canvas.to(torch.float32),
        canvas_mask.to(torch.uint8),
    )

    step_out_path = Path(args.out)
    if args.out_init:
        init_out_path = Path(args.out_init)
    else:
        init_out_path = step_out_path.with_name(f"{step_out_path.stem}_init{step_out_path.suffix}")
    step_out_path.parent.mkdir(parents=True, exist_ok=True)
    init_out_path.parent.mkdir(parents=True, exist_ok=True)

    step_input_names = [
        "frame",
        "ellipse_mask",
        "H_cum_curr",
        "prev_keypoints",
        "prev_descriptors",
        "canvas",
        "canvas_mask",
    ]
    step_output_names = [
        "canvas_out",
        "canvas_mask_out",
        "H_cum_next",
        "prev_keypoints_next",
        "prev_descriptors_next",
        "needs_reset",
    ]
    step_dynamic_axes = {
        "frame": {2: "frame_h", 3: "frame_w"},
        "ellipse_mask": {2: "frame_h", 3: "frame_w"},
        "prev_keypoints": {1: "num_kpts"},
        "prev_descriptors": {1: "num_kpts"},
        "prev_keypoints_next": {1: "num_kpts_next"},
        "prev_descriptors_next": {1: "num_kpts_next"},
    }

    first_wrapper = FirstFrameExportWrapper(inferencer).eval()
    first_inputs = (frame, ellipse_mask)
    first_input_names = ["frame", "ellipse_mask"]
    first_output_names = [
        "canvas",
        "canvas_mask",
        "H_cum_curr",
        "prev_keypoints",
        "prev_descriptors",
    ]
    first_dynamic_axes = {
        "frame": {2: "frame_h", 3: "frame_w"},
        "ellipse_mask": {2: "frame_h", 3: "frame_w"},
        "prev_keypoints": {1: "num_kpts"},
        "prev_descriptors": {1: "num_kpts"},
    }

    torch.onnx.export(
        first_wrapper,
        first_inputs,
        str(init_out_path),
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=first_input_names,
        output_names=first_output_names,
        dynamic_axes=first_dynamic_axes if args.dynamic else None,
    )
    _resave_without_external_data(init_out_path)
    fix_leaked_parameters(init_out_path, first_wrapper, first_input_names)

    torch.onnx.export(
        inferencer,
        step_inputs,
        str(step_out_path),
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=step_input_names,
        output_names=step_output_names,
        dynamic_axes=step_dynamic_axes if args.dynamic else None,
    )
    _resave_without_external_data(step_out_path)
    fix_leaked_parameters(step_out_path, inferencer, step_input_names)

    force_slice_indices_int64(init_out_path)
    force_slice_indices_int64(step_out_path)

    print(f"exported init: {init_out_path}")
    print(f"exported step: {step_out_path}")
    if args.check:
        check_export(init_out_path)
        check_export(step_out_path)
    if args.ort:
        # Free PyTorch model from GPU before ORT smoke test to avoid OOM
        del inferencer, first_wrapper
        import gc; gc.collect()
        torch.cuda.empty_cache()
        ort_smoke_test(init_out_path, first_input_names, first_inputs)
        ort_smoke_test(step_out_path, step_input_names, step_inputs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export models to ONNX.")
    p.add_argument("--out", type=str, required=True, help="Output ONNX file path.")
    p.add_argument("--out-init", type=str, default=None, help="Output ONNX path for first_frame model.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=854)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--ort", action="store_true", help="Run onnxruntime smoke test.")

    p.add_argument("--num-classes", type=int, default=0)
    p.add_argument("--backbone", type=str, default="tu-convnext_base")
    p.add_argument("--seg-weights", type=str, default="weights/fold0.pth")
    p.add_argument("--tool-weights", type=str, default="weights/convnext_tiny-unet-best.pt")
    p.add_argument("--port-weights", type=str, default="weights/convnext_tiny-unet-cholec80_port.pt")
    p.add_argument("--laplacian-var-min", type=float, default=30.0)
    p.add_argument(
        "--stub-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use export-time stubs for ALIKED/LightGlue (disables real feature/matching path).",
    )
    p.add_argument(
        "--register-dcn-symbolic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Register custom ONNX symbolic for torchvision::deform_conv2d.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    export_stitch_inferencer(args)
    step_out_path = Path(args.out)
    if args.out_init:
        init_out_path = Path(args.out_init)
        init_out_fp16_path = init_out_path.with_suffix(".fp16.onnx")
        step_out_fp16_path = step_out_path.with_suffix(".fp16.onnx")
    else:
        init_out_path = step_out_path.with_name(f"{step_out_path.stem}_init{step_out_path.suffix}")
        init_out_fp16_path = init_out_path.with_suffix(".fp16.onnx")
        step_out_fp16_path = step_out_path.with_suffix(".fp16.onnx")
    model_onnx_init = onnx.load(str(init_out_path))
    model_simp_init, check_init = simplify(model_onnx_init)
    if check_init:
        _save_onnx(model_simp_init, init_out_path)

    model_onnx_step = onnx.load(str(step_out_path))
    model_simp_step, check_step = simplify(model_onnx_step)
    if check_step:
        _save_onnx(model_simp_step, step_out_path)

    from onnxconverter_common import float16
    # ORT 1.23 CUDA EP has no FP16 kernel for these ops — keep them in FP32
    # to avoid CPU fallback + Cast cascade (same ops that broke at opset 20).
    fp16_block_list = ["GridSample", "Resize", "AveragePool"]

    model_onnx_init_fp16 = onnx.load(str(init_out_path))
    model_onnx_init_fp16 = float16.convert_float_to_float16(
        model_onnx_init_fp16, op_block_list=fp16_block_list)
    patched = fix_fp16_cast_to_mismatch(model_onnx_init_fp16)
    if patched:
        print(f"patched fp16 Cast(to) mismatch in init model: nodes={patched}")
    model_onnx_init_fp16, check_init = simplify(model_onnx_init_fp16)
    if check_init:
        _save_onnx(model_onnx_init_fp16, init_out_fp16_path)
        force_slice_indices_int64(init_out_fp16_path)

    model_onnx_step_fp16 = onnx.load(str(step_out_path))
    _ensure_node_names(model_onnx_step_fp16)
    hcum_node_names = set(_trace_fp32_nodes(model_onnx_step_fp16, "H_cum_curr", "H_cum_next"))
    print(f"fp16 homography path: {len(hcum_node_names)} nodes")
    model_onnx_step_fp16 = float16.convert_float_to_float16(
        model_onnx_step_fp16, op_block_list=fp16_block_list)
    # Patch homography subgraph back to FP32 after global FP16 conversion.
    _patch_subgraph_to_fp32(model_onnx_step_fp16, hcum_node_names)
    patched = fix_fp16_cast_to_mismatch(model_onnx_step_fp16, skip_prefixes=("_hcum_cast",))
    if patched:
        print(f"patched fp16 Cast(to) mismatch in step model: nodes={patched}")
    # NOTE: Skip onnxsim simplification for the FP16 step model because the
    # simplifier removes the Cast(FP16↔FP32) boundary nodes that protect the
    # homography subgraph from FP16 numerical instability.
    _save_onnx(model_onnx_step_fp16, step_out_fp16_path)
    force_slice_indices_int64(step_out_fp16_path)

    # Final cleanup: strip any non-input entries from graph.input in all files
    init_inputs = ["frame", "ellipse_mask"]
    step_inputs = [
        "frame", "ellipse_mask", "H_cum_curr",
        "prev_keypoints", "prev_descriptors", "canvas", "canvas_mask",
    ]
    for p, names in [
        (init_out_path, init_inputs),
        (step_out_path, step_inputs),
        (init_out_fp16_path, init_inputs),
        (step_out_fp16_path, step_inputs),
    ]:
        if p.exists():
            strip_non_input_graph_inputs(p, names)

if __name__ == "__main__":
    main()
