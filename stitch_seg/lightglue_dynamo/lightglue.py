import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import symbolic_helper
from typing import Protocol


torch.backends.cudnn.deterministic = True
FUSE_MULTI_HEAD_ATTENTION = False
CUSTOM_OP_NAME = "fabiosim::multi_head_attention"
class _OnnxGraphContext(Protocol):
    def op(self, *args: object, **kwargs: object) -> torch._C.Value: ...

def multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = q.shape
    head_dim = d // num_heads
    q, k, v = (t.reshape((b, n, num_heads, head_dim)).transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape((b, n, d))


fused_multi_head_attention = None


def multi_head_attention_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    if FUSE_MULTI_HEAD_ATTENTION and fused_multi_head_attention is not None:
        return fused_multi_head_attention(q, k, v, num_heads)
    else:
        return multi_head_attention(q, k, v, num_heads)


@symbolic_helper.parse_args("v", "v", "v", "i")
def symbolic_multi_head_attention(
    g: _OnnxGraphContext, q: torch._C.Value, k: torch._C.Value, v: torch._C.Value, num_heads_i: int
) -> torch._C.Value:
    return g.op("com.microsoft::MultiHeadAttention", q, k, v, num_heads_i=num_heads_i).setType(q.type())


def use_fused_multi_head_attention() -> None:
    global FUSE_MULTI_HEAD_ATTENTION, fused_multi_head_attention
    FUSE_MULTI_HEAD_ATTENTION = True
    fused_multi_head_attention = torch.library.custom_op(CUSTOM_OP_NAME, mutates_args=())(multi_head_attention)
    torch.onnx.register_custom_op_symbolic(CUSTOM_OP_NAME, symbolic_multi_head_attention, 9)


class LearnableFourierPositionalEncoding(nn.Module):
    num_heads: int
    gamma: float

    def __init__(self, M: int, descriptor_dim: int, num_heads: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.num_heads = num_heads  # type: ignore[unresolved-attribute]
        head_dim = descriptor_dim // num_heads
        self.Wr = nn.Linear(M, head_dim // 2, bias=False)
        self.gamma = gamma  # type: ignore[unresolved-attribute]
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines])
        return emb.repeat_interleave(2, dim=3).repeat(1, 1, 1, self.num_heads).unsqueeze(4)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """get confidence tokens"""
        return (self.token(desc0.detach()).squeeze(-1), self.token(desc1.detach()).squeeze(-1))


class SelfBlock(nn.Module):
    embed_dim: int
    num_heads: int
    head_dim: int

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim  # type: ignore[unresolved-attribute]
        self.num_heads = num_heads  # type: ignore[unresolved-attribute]
        self.head_dim = embed_dim // num_heads  # type: ignore[unresolved-attribute]
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv: torch.Tensor = self.Wqkv(x)
        qkv = qkv.reshape((b, n, self.embed_dim, 3))
        qk, v = qkv[..., :2], qkv[..., 2]
        qk = self.apply_cached_rotary_emb(encoding, qk)
        q, k = qk[..., 0], qk[..., 1]
        context = multi_head_attention_dispatch(q, k, v, self.num_heads)
        message = self.out_proj(context)
        return x + self.ffn(torch.concat([x, message], 2))

    def rotate_half(self, qk: torch.Tensor) -> torch.Tensor:
        b, n, _, _ = qk.shape
        qk = qk.reshape((b, n, self.num_heads, self.head_dim // 2, 2, 2))
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=4)
        qk = qk.reshape((b, n, self.embed_dim, 2))
        return qk

    def apply_cached_rotary_emb(self, encoding: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
        return qk * encoding[0] + self.rotate_half(qk) * encoding[1]


class CrossBlock(nn.Module):
    embed_dim: int
    num_heads: int
    head_dim: int

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim  # type: ignore[unresolved-attribute]
        self.num_heads = num_heads  # type: ignore[unresolved-attribute]
        self.head_dim = embed_dim // num_heads  # type: ignore[unresolved-attribute]
        self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        b, _, _ = descriptors.shape
        qk, v = self.to_qk(descriptors), self.to_v(descriptors)

        indices = torch.arange(b, device=descriptors.device)
        swap = (indices // 2) * 2 + (1 - indices % 2)  # swap trick
        m = multi_head_attention_dispatch(qk, qk[swap], v[swap], self.num_heads)
        m = self.to_out(m)
        descriptors = descriptors + self.ffn(torch.concat([descriptors, m], 2))
        return descriptors


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(self, descriptors: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
        descriptors = self.self_attn(descriptors, encodings)
        return self.cross_attn(descriptors)


def sigmoid_log_double_softmax(similarities: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    certainties = F.logsigmoid(z[0::2]) + F.logsigmoid(z[1::2]).transpose(1, 2)
    scores0 = F.log_softmax(similarities, 2)
    scores1 = F.log_softmax(similarities, 1)
    scores = scores0 + scores1 + certainties
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.matchability = nn.Linear(dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """build assignment matrix from descriptors"""
        mdescriptors = self.final_proj(descriptors) / self.scale
        similarities = mdescriptors[0::2] @ mdescriptors[1::2].transpose(1, 2)
        z = self.matchability(descriptors)
        scores = sigmoid_log_double_softmax(similarities, z)
        return scores

    def get_matchability(self, desc: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """obtain matches from a log assignment matrix [BxNxN]

    Returns fixed-size (N, 2) matches and (N,) scores.
    Invalid entries have score=0 and safe placeholder indices.
    This avoids NonZero/torch.where which is slow in ONNX Runtime CUDA.

    IMPORTANT: All index tensors are derived from existing CUDA tensors
    (indices, m0) to avoid ConstantOfShape/Range ops that ORT CUDA EP
    does not support, which would cause cascading CPU fallback.
    """
    max0 = scores.max(2)
    max1 = scores.max(1)
    m0, m1 = max0.indices, max1.indices

    indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
    mutual = indices == m1.gather(1, m0)
    mscores = max0.values.exp()
    valid = (mscores > threshold) & mutual  # [B, N]

    # Fixed-size output (B=1 assumed for ONNX export)
    valid_f = valid[0].float()  # [N]
    out_scores = mscores[0] * valid_f  # 0 for invalid

    # Reuse existing CUDA tensors to avoid ConstantOfShape (CPU fallback)
    m0_idx = indices[0]  # [N], already on CUDA from arange above
    # For invalid matches set index to 0 (safe for gather; weight=0 handles correctness)
    m1_idx = m0[0] * valid[0].long()

    matches = torch.stack([m0_idx, m1_idx], dim=1)  # [N, 2]
    return matches, out_scores


class LightGlue(nn.Module):
    descriptor_dim: int
    num_heads: int
    n_layers: int
    filter_threshold: float
    depth_confidence: float
    width_confidence: float
    confidence_thresholds: torch.Tensor

    def __init__(
        self,
        weights: str,
        input_dim: int = 128,
        descriptor_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 9,
        filter_threshold: float = 0.1,  # match threshold
        depth_confidence: float = -1,  # -1 is no early stopping, recommend: 0.95
        width_confidence: float = -1,  # -1 is no point pruning, recommend: 0.99
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, descriptor_dim, bias=True)
        self.descriptor_dim = descriptor_dim  # type: ignore[unresolved-attribute]
        self.num_heads = num_heads  # type: ignore[unresolved-attribute]
        self.n_layers = n_layers  # type: ignore[unresolved-attribute]
        self.filter_threshold = filter_threshold  # type: ignore[unresolved-attribute]
        self.depth_confidence = depth_confidence  # type: ignore[unresolved-attribute]
        self.width_confidence = width_confidence  # type: ignore[unresolved-attribute]

        self.posenc = LearnableFourierPositionalEncoding(2, self.descriptor_dim, self.num_heads)

        d, h, n = self.descriptor_dim, self.num_heads, self.n_layers

        self.transformers = nn.ModuleList([TransformerLayer(d, h) for _ in range(n)])

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        state_dict = torch.load(weights, map_location="cpu")
        

        # rename old state dict entries
        for i in range(n):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        keypoints: torch.Tensor,  # (2B, N, 2), normalized
        descriptors: torch.Tensor,  # (2B, N, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        descriptors = self.input_proj(descriptors)
        # positional embeddings
        encodings = self.posenc(keypoints)  # (2, 2B, *, 64, 1)

        # GNN + final_proj + assignment
        for i in range(self.n_layers):
            # self+cross attention
            descriptors = self.transformers[i](descriptors, encodings)

        scores = self.log_assignment[i](descriptors)  # (B, N, N)
        matches, mscores = filter_matches(scores, self.filter_threshold)
        return matches, mscores  # (M, 3), (M,)

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor | None, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        return keep

    def check_if_stop(
        self, confidences0: torch.Tensor, confidences1: torch.Tensor, layer_index: int, num_points: int
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence