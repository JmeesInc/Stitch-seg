#!/usr/bin/env python3
"""ONNX export utilities for segmentation model / stitch inferencer."""
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


import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
import cumprod_onnx_exporter
cumprod_onnx_exporter.register_cumprod_onnx_op()


class FirstFrameExportWrapper(nn.Module):
    """Wrapper to export inferencer.first_frame as a standalone ONNX graph."""

    def __init__(self, inferencer):
        super().__init__()
        self.inferencer = inferencer

    def forward(self, frame: torch.Tensor, ellipse_mask: torch.Tensor):
        canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors = self.inferencer.first_frame(frame, ellipse_mask)
        return canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors


class CanvasSegExportModel(nn.Module):
    """Segmentation model used by the stitch inferencer."""

    def __init__(self, cfg, height, width):
        super().__init__()
        num_classes = int(getattr(cfg, "num_classes", 1))
        backbone = getattr(cfg, "backbone", "tu-convnext_base")
        self.height = height
        self.width = width
        self.new_h = ((height + 31) // 32) * 32
        self.new_w = ((width + 31) // 32) * 32
        self._using_fallback = False
        from model import UnetPlusPlus

        self.model = UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        ).to(cfg.device)

        self.model.load_state_dict(torch.load(cfg.segmentation_weights, map_location=cfg.device), strict=True)
        self.model.eval()
    
    def forward(self, image_u8: torch.Tensor) -> torch.Tensor:
        x = image_u8.float() / 255.0
        x_model = F.interpolate(x, size=(self.new_h, self.new_w), mode="bilinear", align_corners=False)

        y = self.model(x_model)
        y = F.interpolate(y, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return y



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

        for inp_idx in (1, 2):  # starts, ends
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
        onnx.save(model, str(path))
        print(f"patched Slice index dtype to int64: {path} (nodes={changed})")


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


def fix_fp16_cast_to_mismatch(model: onnx.ModelProto) -> int:
    """Fix known onnxconverter-common FP16 conversion issue.

    `float16.convert_float_to_float16()` sometimes updates `value_info` to FP16 but
    leaves `Cast(to=FLOAT)` nodes unchanged, causing ORT type validation failures.
    We patch those Cast nodes to `to=FLOAT16` when their outputs are annotated FP16.
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
        if any(vi_elem_type.get(out) == TensorProto.FLOAT16 for out in node.output):
            to_attr.i = int(TensorProto.FLOAT16)
            patched += 1
    return patched


def export_stitch_inferencer(args: argparse.Namespace):
    from stitch_seg.inferencer_toonnx import StitchInferencer_ONNX

    cfg = build_cfg(args)
    seg_model = CanvasSegExportModel(cfg, args.height, args.width).eval()
    inferencer = StitchInferencer_ONNX(seg_model, cfg=cfg, input_size=(args.height, args.width)).eval()

    h, w = args.height, args.width
    frame = torch.randint(0, 256, (1, 3, h, w), device=cfg.device, dtype=torch.float32)
    ellipse_mask = torch.zeros((1, 1, h, w), device=cfg.device, dtype=torch.float32)
    with torch.inference_mode():
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
        "seg_map",
        "canvas_out",
        "canvas_mask_out",
        "H_cum_next",
        "prev_keypoints_next",
        "prev_descriptors_next",
    ]
    step_dynamic_axes = {
        "frame": {2: "frame_h", 3: "frame_w"},
        "ellipse_mask": {2: "frame_h", 3: "frame_w"},
        "prev_keypoints": {1: "num_kpts"},
        "prev_descriptors": {1: "num_kpts"},
        "seg_map": {1: "frame_h", 2: "frame_w"},
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

    with torch.inference_mode():
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

    with torch.inference_mode():
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

    force_slice_indices_int64(init_out_path)
    force_slice_indices_int64(step_out_path)

    print(f"exported init: {init_out_path}")
    print(f"exported step: {step_out_path}")
    if args.check:
        check_export(init_out_path)
        check_export(step_out_path)
    if args.ort:
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

    p.add_argument("--num-classes", type=int, default=13)
    p.add_argument("--backbone", type=str, default="tu-convnext_base")
    p.add_argument("--seg-weights", type=str, default="weights/fold0.pth")
    p.add_argument("--tool-weights", type=str, default="weights/convnext_tiny-unet-best.pt")
    p.add_argument("--port-weights", type=str, default="weights/convnext_tiny-unet-cholec80_port.pt")
    p.add_argument("--laplacian-var-min", type=float, default=60.0)
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
    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
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
        onnx.save(model_simp_init, str(init_out_path))

    model_onnx_step = onnx.load(str(step_out_path))
    model_simp_step, check_step = simplify(model_onnx_step)
    if check_step:
        onnx.save(model_simp_step, str(step_out_path))
    
    from onnxconverter_common import float16
    model_onnx_init_fp16 = onnx.load(str(init_out_path))
    model_onnx_init_fp16 = float16.convert_float_to_float16(model_onnx_init_fp16)
    patched = fix_fp16_cast_to_mismatch(model_onnx_init_fp16)
    if patched:
        print(f"patched fp16 Cast(to) mismatch in init model: nodes={patched}")
    model_onnx_init_fp16, check_init = simplify(model_onnx_init_fp16)
    if check_init:
        onnx.save(model_onnx_init_fp16, str(init_out_fp16_path))

    model_onnx_step_fp16 = onnx.load(str(step_out_path))
    model_onnx_step_fp16 = float16.convert_float_to_float16(model_onnx_step_fp16)
    patched = fix_fp16_cast_to_mismatch(model_onnx_step_fp16)
    if patched:
        print(f"patched fp16 Cast(to) mismatch in step model: nodes={patched}")
    model_onnx_step_fp16, check_step = simplify(model_onnx_step_fp16)
    if check_step:
        onnx.save(model_onnx_step_fp16, str(step_out_fp16_path))

if __name__ == "__main__":
    main()
