"""
This exporter requires opset version 12 to support the following operators:
  - Clip:
    It can accept tensor(int64) from version 12.
  - GatherND:
    It can support batch_dims from version 12.
"""

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help
try:
    from torch.onnx._type_utils import JitScalarType
except ImportError:
    JitScalarType = None

__all__ = ["register_cumprod_onnx_op"]

onnx_opset_version = 13


def _resolve_input_dtype(g, x):
    if JitScalarType is not None and hasattr(JitScalarType, "from_value"):
        try:
            st = JitScalarType.from_value(x)
            return st.onnx_type()
        except Exception:
            pass
    scalar_type = sym_help._try_get_scalar_type(x)
    if scalar_type in sym_help.cast_pytorch_to_onnx:
        return sym_help.cast_pytorch_to_onnx[scalar_type]
    return None


def cumprod_symbolic(g, input, dim, dtype=None):
    """
    aten::cumprod (A.cumprod() または torch.cumprod()) を
    Log -> CumSum -> Exp に置き換える変換関数。
    """
    # 1. dim (軸) の値を取得
    # PyTorchのトレース時はTensorや定数として渡ってくるため、ヘルパーでintを取り出す
    dim_val = sym_help._get_const(dim, 'i', 'dim')

    # Cumprod for bool/int is used in this project; Log requires float.
    input_onnx_dtype = _resolve_input_dtype(g, input)
    need_cast_in = input_onnx_dtype in (
        sym_help.cast_pytorch_to_onnx.get("Bool"),
        sym_help.cast_pytorch_to_onnx.get("Long"),
        sym_help.cast_pytorch_to_onnx.get("Int"),
        sym_help.cast_pytorch_to_onnx.get("Short"),
        sym_help.cast_pytorch_to_onnx.get("Byte"),
    )
    x = g.op("Cast", input, to_i=sym_help.cast_pytorch_to_onnx["Float"]) if need_cast_in else input

    # 2. Log(x)
    # 0が含まれると -inf になりますが、後のExpで 0 に戻るため
    # 「0以上の値」であれば数学的に問題なく動作します。
    log_x = g.op("Log", x)

    # 3. CumSum(log(x))
    # ONNXのCumSumは axis を属性(attribute)ではなく入力(input)として受け取ります
    # (Opset 11以降の仕様)
    axis_tensor = g.op("Constant", value_t=torch.tensor(dim_val, dtype=torch.long))
    cumsum_log = g.op("CumSum", log_x, axis_tensor)

    # 4. Exp(cumsum) -> 元の空間に戻す
    output = g.op("Exp", cumsum_log)

    # Respect explicit dtype argument if provided.
    if dtype is not None and not sym_help._is_none(dtype):
        dtype_const = sym_help._get_const(dtype, "i", "dtype")
        target = sym_help.cast_pytorch_to_onnx[sym_help.scalar_type_to_pytorch_type[dtype_const].__name__]
        return g.op("Cast", output, to_i=target)

    # Otherwise cast back to original type when input was non-float.
    if need_cast_in and input_onnx_dtype is not None:
        return g.op("Cast", output, to_i=input_onnx_dtype)
    return output

def register_cumprod_onnx_op():
    register_custom_op_symbolic('aten::cumprod', cumprod_symbolic, 11)
