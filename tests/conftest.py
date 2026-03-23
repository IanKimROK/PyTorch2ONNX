"""
Shared pytest fixtures for PyTorch2ONNX test suite.

All tests use a tiny two-layer CNN (TinyModel) instead of downloading
real torchvision weights, so the suite runs fast and offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny dummy model (avoids network downloads in CI)
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal CNN: conv → relu → adaptive-pool → linear."""

    IN_CHANNELS = 3
    NUM_CLASSES = 10

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        return self.classifier(x.flatten(1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model() -> TinyModel:
    """Return a fresh TinyModel in eval mode."""
    model = TinyModel()
    model.eval()
    return model


@pytest.fixture(scope="session")
def input_shape() -> tuple[int, int, int, int]:
    return (1, 3, 32, 32)


@pytest.fixture(scope="session")
def dummy_input(input_shape) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(*input_shape)


@pytest.fixture(scope="session")
def dummy_array(dummy_input) -> np.ndarray:
    return dummy_input.numpy()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for model files."""
    return tmp_path


@pytest.fixture(scope="session")
def tiny_onnx_path(tiny_model, input_shape, tmp_path_factory) -> str:
    """Export TinyModel to a temp ONNX file once per session."""
    out = tmp_path_factory.mktemp("models") / "tiny.onnx"
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        tiny_model,
        dummy,
        str(out),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return str(out)
