#!/usr/bin/env python3
"""Export CanvasSegExportModel (UPerNet segmentation only) to ONNX."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxsim import simplify
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# AdaptiveAvgPool2d patch (same as export_onnx.py)
# ---------------------------------------------------------------------------

class _OnnxAdaptiveAvgPool2d(nn.Module):
    """Replace AdaptiveAvgPool2d with F.interpolate for ONNX export."""

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)


def _patch_adaptive_avg_pool2d(model: nn.Module) -> None:
    for name, child in model.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(model, name, _OnnxAdaptiveAvgPool2d(child.output_size))
        else:
            _patch_adaptive_avg_pool2d(child)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CanvasSegExportModel(nn.Module):
    """Segmentation model matching the original CanvasSegModel interface."""

    def __init__(self, cfg):
        super().__init__()
        num_classes = int(getattr(cfg, "num_classes", 13))
        backbone = "tu-hrnet_w32"
        self.input_size = int(getattr(cfg, "image_size", 512))
        '''from model import UnetPlusPlus
        self.model = UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        ).to(cfg.device)'''

        self.model.load_state_dict(
            torch.load("models/hrnet/fold0.pth", map_location=cfg.device), strict=True
        )
        _patch_adaptive_avg_pool2d(self.model)
        self.model.eval()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(cfg.device)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(cfg.device)
        )

    def forward(self, image_u8: torch.Tensor) -> torch.Tensor:
        x = image_u8.float() / 255.0
        x = (x - self.mean) / self.std
        target_hw = x.shape[-2:]
        x_model = F.interpolate(
            x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )
        y = self.model(x_model)
        y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        return y


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export seg-only model to ONNX.")
    p.add_argument("--out", type=str, default="onnx/seg.onnx", help="Output ONNX file path.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=854)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--check", action="store_true")
    p.add_argument("--ort", action="store_true", help="Run onnxruntime smoke test.")
    p.add_argument("--num-classes", type=int, default=13)
    p.add_argument("--backbone", type=str, default="tu-convnext_base")
    p.add_argument("--image-size", type=int, default=512,
                   help="Internal model resolution (default: 512).")
    p.add_argument("--seg-weights", type=str, default="models/upernet/fold0.pth")
    p.add_argument("--fp16", action="store_true", help="Also export FP16 variant.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = SimpleNamespace(
        device=device,
        num_classes=args.num_classes,
        backbone=args.backbone,
        image_size=args.image_size,
        segmentation_weights=args.seg_weights,
    )

    model = CanvasSegExportModel(cfg).eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = args.height, args.width
    dummy = torch.randint(0, 256, (1, 3, h, w), device=device, dtype=torch.float32)

    input_names = ["image"]
    output_names = ["logits"]
    dynamic_axes = {
        "image": {2: "H", 3: "W"},
        "logits": {2: "H", 3: "W"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print(f"exported: {out_path}")

    # Simplify
    model_onnx = onnx.load(str(out_path))
    model_simp, ok = simplify(model_onnx)
    if ok:
        onnx.save(model_simp, str(out_path))
        print(f"simplified: {out_path}")

    if args.check:
        onnx.checker.check_model(onnx.load(str(out_path)))
        print(f"onnx checker passed: {out_path}")

    if args.ort:
        sess = ort.InferenceSession(str(out_path), providers=["CUDAExecutionProvider"])
        ort_inputs = {"image": dummy.detach().cpu().numpy()}
        _ = sess.run(None, ort_inputs)
        print(f"onnxruntime smoke test passed: {out_path}")

    # FP16
    if args.fp16:
        from onnxconverter_common import float16

        fp16_path = out_path.with_suffix(".fp16.onnx")
        fp16_block_list = ["GridSample", "Resize", "AveragePool"]
        model_fp16 = onnx.load(str(out_path))
        model_fp16 = float16.convert_float_to_float16(
            model_fp16, op_block_list=fp16_block_list
        )
        model_fp16, ok = simplify(model_fp16)
        if ok:
            onnx.save(model_fp16, str(fp16_path))
            print(f"exported fp16: {fp16_path}")


if __name__ == "__main__":
    main()
