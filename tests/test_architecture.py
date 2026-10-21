"""Minimal architecture smoke test; no data or checkpoint is required."""

import unittest

import torch

from subfusion.model import SubFusionUNet


class ArchitectureTest(unittest.TestCase):
    def test_two_input_unet_output_shape(self) -> None:
        model = SubFusionUNet().eval()
        with torch.inference_mode():
            fused = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(fused.shape), (1, 3, 64, 64))
