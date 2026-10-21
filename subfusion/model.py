"""Generator-only, two-input U-Net used by SubFusion."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .blocks import Center2D, Decoder2D, Encoder2D, First2D, Last2D


def pad_to_shape(tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Right/bottom-pad a decoder feature map to the corresponding skip shape."""
    return functional.pad(tensor, (0, shape[3] - tensor.shape[3], 0, shape[2] - tensor.shape[2]))


class SubFusionUNet(nn.Module):
    """Visible/IR U-Net with independent encoders and learned scale-wise fusion.

    The visible RGB and replicated-RGB infrared tensors are processed by
    separate encoders.  At each scale, concatenated source features are
    projected by a 1×1 convolution before entering the shared decoder.
    """

    def __init__(self, input_channels: int = 3, output_channels: int = 3):
        super().__init__()
        depths = (64, 128, 256, 512, 1024)
        self.encoder_layers = nn.Sequential(
            First2D(input_channels, depths[0], depths[0]),
            *[Encoder2D(depths[index], depths[index + 1], depths[index + 1]) for index in range(len(depths) - 2)],
        )
        self.encoder_layers2 = nn.Sequential(
            First2D(input_channels, depths[0], depths[0]),
            *[Encoder2D(depths[index], depths[index + 1], depths[index + 1]) for index in range(len(depths) - 2)],
        )
        self.fusion_layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for channels in depths[:-1]
        )
        self.center = Center2D(depths[-2], depths[-1], depths[-1], depths[-2])
        self.decoder_layers = nn.Sequential(
            *[
                Decoder2D(2 * depths[index + 1], 2 * depths[index], 2 * depths[index], depths[index])
                for index in reversed(range(len(depths) - 2))
            ],
            Last2D(depths[1], depths[0], output_channels),
        )

    def forward(self, visible: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        visible_features, infrared_features = [], []
        visible_current, infrared_current = visible, infrared
        for visible_encoder, infrared_encoder in zip(self.encoder_layers, self.encoder_layers2):
            visible_current = visible_encoder(visible_current)
            infrared_current = infrared_encoder(infrared_current)
            visible_features.append(visible_current)
            infrared_features.append(infrared_current)
        fused_features = [
            fusion(torch.cat((visible_feature, infrared_feature), dim=1))
            for visible_feature, infrared_feature, fusion in zip(visible_features, infrared_features, self.fusion_layers)
        ]
        decoded = [self.center(fused_features[-1])]
        for layer_index, decoder in enumerate(self.decoder_layers):
            skip = fused_features[-1 - layer_index]
            decoded.append(decoder(torch.cat((pad_to_shape(decoded[-1], skip.shape), skip), dim=1)))
        return decoded[-1]


def load_msrs_checkpoint(path: str, device: torch.device | str = "cpu") -> SubFusionUNet:
    """Load the distributed MSRS rank-1 checkpoint with strict key checking."""
    model = SubFusionUNet()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument
        state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
