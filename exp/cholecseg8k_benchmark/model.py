import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder


def _infer_effective_depth_from_encoder_channels(encoder_channels: Sequence[int]) -> int:
    """
    segmentation_models_pytorch's `TimmUniversalEncoder` inserts a "dummy feature (C=0)"
    for transformer-style backbones (e.g. ConvNeXt) where 1/2-resolution features do not exist,
    in order to align the feature list.

    The Unet++ decoder implementation has a code path that uses `skip_channels` as out_channels,
    so if skip_channels=0 is present it will create a Conv2d with out_channels=0 and crash at runtime.

    This function returns the effective depth (= number of stages) after excluding C=0 stages
    from the decoder.
    """
    if len(encoder_channels) < 2:
        return 0
    # index 0 is the input (in_channels) and is always kept; drop any subsequent stage where C==0.
    non_zero_stages = [c for c in encoder_channels[1:] if c > 0]
    return len(non_zero_stages)


def _filter_zero_channel_features(
    features: Sequence[torch.Tensor], encoder_channels: Sequence[int]
) -> list[torch.Tensor]:
    # features and encoder_channels are assumed to be in the same order (scale 1, 1/2, 1/4, ...)
    if len(features) != len(encoder_channels):
        # Sanity check. If this is broken, subsequent consistency cannot be guaranteed, so fail immediately.
        raise RuntimeError(
            f"encoder returned {len(features)} feature maps, but encoder.out_channels has {len(encoder_channels)} entries"
        )
    kept = [features[0]]  # input
    for feat, ch in zip(features[1:], encoder_channels[1:]):
        if ch > 0:
            kept.append(feat)
    return kept


def _filter_zero_channels(encoder_channels: Sequence[int]) -> list[int]:
    kept = [encoder_channels[0]]
    kept.extend([c for c in encoder_channels[1:] if c > 0])
    return kept


class UnetPlusPlus(SegmentationModel):
    """
    A fixed Unet++ that avoids the crash in `smp.UnetPlusPlus` when using
    `tu-*` (TimmUniversalEncoder) transformer-style backbones (e.g. `tu-convnext_base`).

    Root cause:
    - `TimmUniversalEncoder` pads missing 1/2-scale features with dummies (C=0)
      (inserts 0 into out_channels in `encoders/timm_universal.py` and inserts C=0 tensors in forward)
    - `UnetPlusPlusDecoder` uses `out_channels = skip_channels` in some blocks, so
      skip_channels=0 produces a Conv2d with weight shape [0, ...] which raises an error at runtime

    Fix:
    - Exclude C=0 stages from the encoder features and out_channels, and build the decoder
      one stage shallower (for ConvNeXt etc. this effectively gives a 4-stage Unet++ at 1/4~1/32)
    """

    _is_torch_scriptable = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError(f"UnetPlusPlus is not support encoder_name={encoder_name}")

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "decoder_use_batchnorm is deprecated. Use decoder_use_norm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        # Exclude C=0 stages to determine encoder_channels / depth passed to the decoder
        full_encoder_channels = list(self.encoder.out_channels)
        effective_depth = _infer_effective_depth_from_encoder_channels(full_encoder_channels)
        if effective_depth <= 0:
            raise RuntimeError(
                f"effective_depth is {effective_depth}; encoder.out_channels={full_encoder_channels}"
            )
        filtered_encoder_channels = _filter_zero_channels(full_encoder_channels)

        # Use only the first effective_depth entries from the (originally specified) decoder_channels
        decoder_channels = tuple(decoder_channels)[:effective_depth]
        if len(decoder_channels) != effective_depth:
            raise ValueError(
                f"effective_depth={effective_depth} but decoder_channels has length {len(decoder_channels)}."
                f" encoder.out_channels={full_encoder_channels}"
            )

        self._full_encoder_channels = full_encoder_channels

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=filtered_encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=effective_depth,
            use_norm=decoder_use_norm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=filtered_encoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"unetplusplus-fixed-{encoder_name}"
        self.initialize()

    def forward(self, x: torch.Tensor):
        # Follow SegmentationModel.forward while dropping C=0 stage features before passing to the decoder
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self.check_input_shape(x)

        features = self.encoder(x)
        features = _filter_zero_channel_features(features, self._full_encoder_channels)

        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        # Restore to the same resolution as the input (for consistency in mask supervision / metric computation)
        if masks.shape[-2:] != x.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks


# ==========================================
# 1. SP-TCN Components (Defined in previous step)
# ==========================================

class DilatedResidualLayer(nn.Module):
    """
    3D Dilated Residual Layer.
    Ref: Section 3.1, Figure 2 [cite: 110, 112]
    """
    def __init__(self, channels, kernel_size, dilation):
        super(DilatedResidualLayer, self).__init__()
        kt, kh, kw = kernel_size
        dt = dilation
        pad_t = int(dt * (kt - 1) / 2)
        pad_h = int((kh - 1) / 2)
        pad_w = int((kw - 1) / 2)
        
        self.w_norm_dilated_conv = weight_norm(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, 
                      padding=(pad_t, pad_h, pad_w), dilation=(dt, 1, 1))
        ) # Weighted normalized Dilated Conv
        self.bn = nn.BatchNorm3d(channels) # SyncBatchNorm - we don't use multi-GPU training so BatchNorm is enough
        self.relu = nn.ReLU(inplace=True)
        
        std_pad_t = int((kt - 1) / 2)
        self.final_conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, 
                                    padding=(std_pad_t, pad_h, pad_w))

    def forward(self, x):
        residual = x
        out = self.w_norm_dilated_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.final_conv(out)
        return out + residual

class SegmentationLayer(nn.Module):
    """
    Segmentation Layer.
    Ref: Figure 2 [cite: 82]
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SegmentationLayer, self).__init__()
        pad_t = int((kernel_size[0] - 1) / 2)
        padding = (pad_t, int((kernel_size[1]-1)/2), int((kernel_size[2]-1)/2))
        
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class SPTCN(nn.Module):
    """
    Spatial Temporal Convolutional Network (Decoder).
    Ref: Section 3.1, Figure 2 [cite: 87, 95]
    """
    def __init__(self, in_channels, num_classes, num_layers=4, feature_dim=128, time_kernel_size=3):
        super(SPTCN, self).__init__()
        kernel_size = (time_kernel_size, 3, 3)
        pad_t = int((time_kernel_size - 1) / 2)
        
        self.initial_conv = nn.Conv3d(in_channels, feature_dim, kernel_size, padding=(pad_t, 1, 1))
        
        layers = []
        for i in range(num_layers):
            layers.append(DilatedResidualLayer(feature_dim, kernel_size, dilation=2**i))
        self.residual_layers = nn.Sequential(*layers)
        
        self.mid_conv = nn.Conv3d(feature_dim, feature_dim, kernel_size, padding=(pad_t, 1, 1))
        self.segmentation_layer = SegmentationLayer(feature_dim, num_classes, kernel_size)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_layers(x)
        x = self.mid_conv(x)
        x = self.segmentation_layer(x)
        return x

# ==========================================
# 2. Encoder Wrapper (timm Integration)
# ==========================================

class EncoderWrapper(nn.Module):
    """
    Wraps timm models (HRNet, Swin) to process video frames.
    
    The paper states: "An encoder processes individual frames whilst the decoder
    processes a temporal batch of adjacent frames." [cite: 13]
    """
    def __init__(self, model_name:str="hrnet_w32", pretrained:bool=True):
        super(EncoderWrapper, self).__init__()
        
        # Load model from timm with features_only=True to get feature maps
        # HRNet32: 'hrnet_w32'
        # Swin Base: 'swin_base_patch4_window7_224' (example configuration)
        self.encoder = get_encoder(model_name, in_channels=3, weights="imagenet")
        
        # Get the number of channels of the last feature map
        feature_info = self.encoder.model.feature_info
        self.out_channels = feature_info[-1]['num_chs']
        
        print(f"Loaded encoder: {model_name}")
        print(f"Encoder output channels (input to SP-TCN): {self.out_channels}")

    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Time, Height, Width) - Video Batch
        Returns:
            features: (Batch, Out_Channels, Time, H', W')
        """
        B, C, T, H, W = x.shape
        
        # Fold Time into Batch dimension to process frames individually 
        # x -> (B*T, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # Pass through encoder
        # timm returns a list of feature maps. We take the last one.
        features_list = self.encoder(x_reshaped)
        last_feature = features_list[-1] 
        
        # Unfold Time dimension
        # out -> (B, T, C', H', W') -> Permute to (B, C', T, H', W') for SP-TCN (Conv3D)
        _, C_out, H_out, W_out = last_feature.shape
        features = last_feature.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        
        return features

# ==========================================
# 3. Full Video Segmentation Model
# ==========================================

class SPTCN_TemporalSegmentation(nn.Module):
    """
    Complete architecture: Encoder + SP-TCN Decoder.
    Supports HRNet32 and Swin Base as described in Section 4.
    """
    def __init__(self, encoder_name:str="tu-hrnet_w32", num_classes:int=13, num_layers:int=4, feature_dim:int=128):
        super(SPTCN_TemporalSegmentation, self).__init__()
        
        # 1. Encoder
        self.encoder = EncoderWrapper(encoder_name)
        
        # 2. Temporal Decoder (SP-TCN)
        # Input channels must match encoder output
        self.decoder = SPTCN(
            in_channels=self.encoder.out_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            feature_dim=feature_dim
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, 3, Time, Height, Width) - Normalized image sequence
        Returns:
            logits: (Batch, Num_Classes, Time, Height, Width) - Upsampled segmentation map
        """
        input_size = x.shape[-2:] # (H, W)
        
        # Extract spatio-temporal features
        # [z_t] = E([x]_t) [cite: 95]
        features = self.encoder(x)
        
        # Decode features to segmentation logits
        # [S_t] = D([z_t]) [cite: 93]
        logits = self.decoder(features)
        
        # Upsample to original input resolution
        # Usually required as encoder downsamples the image
        logits = F.interpolate(
            logits, 
            size=(x.shape[2], input_size[0], input_size[1]), # (T, H, W)
            mode='trilinear', 
            align_corners=False
        )
        
        return logits