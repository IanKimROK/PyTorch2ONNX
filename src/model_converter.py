"""
Model Converter Module
Handles PyTorch to ONNX conversion with dynamic batch size support.

Supports two export paths:
  - Legacy  : torch.onnx.export()         (stable, opset-based)
  - Dynamo  : torch.onnx.export(dynamo=True)  (PyTorch 2.x, graph-capture)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.onnx
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for PyTorch → ONNX conversion."""

    model_name: str = "resnet18"
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
    pretrained: bool = True
    opset_version: int = 17          # ONNX opset 17 is current stable (≥ ORT 1.16)
    use_dynamo: bool = False         # Use torch.onnx.export(dynamo=True) when True
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])


class ModelConverter:
    """Convert PyTorch models to ONNX format with dynamic batch size support."""

    def __init__(self, config: ConversionConfig | None = None, **kwargs):
        """
        Initialize the converter.

        Args:
            config: ConversionConfig dataclass.  If None, one is built from kwargs
                    for backwards compatibility (model_name, input_shape, pretrained).
        """
        if config is None:
            config = ConversionConfig(
                model_name=kwargs.get("model_name", "resnet18"),
                input_shape=kwargs.get("input_shape", (1, 3, 224, 224)),
                pretrained=kwargs.get("pretrained", True),
            )
        self.config = config
        self.pytorch_model = self._load_pytorch_model()
        self.pytorch_model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pytorch_model(self) -> torch.nn.Module:
        """
        Load a torchvision model by name using the modern weights API.

        torchvision ≥ 0.13 introduced the ``weights=`` parameter and deprecated
        ``pretrained=True``.  We use ``tv_models.get_model`` +
        ``tv_models.get_model_weights`` so the same code path handles *all*
        architectures uniformly.
        """
        name = self.config.model_name
        if not hasattr(tv_models, name):
            raise ValueError(
                f"Model '{name}' not found in torchvision.models. "
                f"Available models: {tv_models.list_models()[:10]} …"
            )

        if self.config.pretrained:
            try:
                # get_model_weights returns a WeightsEnum; .DEFAULT is the best
                weights_enum = tv_models.get_model_weights(name)
                weights = weights_enum.DEFAULT
            except Exception:
                # Fallback: some custom / contrib models may not have a weights enum
                logger.warning(
                    "Could not resolve weights enum for '%s'; loading without pretrained weights.",
                    name,
                )
                weights = None
        else:
            weights = None

        model = tv_models.get_model(name, weights=weights)
        logger.info("Loaded model '%s' (pretrained=%s)", name, self.config.pretrained)
        return model

    def _make_dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        shape = list(self.config.input_shape)
        shape[0] = batch_size
        return torch.randn(*shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_to_onnx(self, output_path: str) -> str:
        """
        Export the PyTorch model to ONNX.

        Chooses between the legacy ``torch.onnx.export`` and the newer
        dynamo-based exporter depending on ``config.use_dynamo``.

        Args:
            output_path: Destination ``.onnx`` file path.

        Returns:
            Absolute path to the saved ONNX model.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dummy_input = self._make_dummy_input()

        if self.config.use_dynamo:
            self._export_dynamo(dummy_input, output_path)
        else:
            self._export_legacy(dummy_input, output_path)

        logger.info("Exported '%s' → %s", self.config.model_name, output_path)
        return output_path

    def _export_legacy(self, dummy_input: torch.Tensor, output_path: str) -> None:
        """Classic torch.onnx.export path (opset-based, stable)."""
        dynamic_axes = {
            self.config.input_names[0]: {0: "batch_size"},
            self.config.output_names[0]: {0: "batch_size"},
        }
        torch.onnx.export(
            self.pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=self.config.input_names,
            output_names=self.config.output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

    def _export_dynamo(self, dummy_input: torch.Tensor, output_path: str) -> None:
        """
        Dynamo-based exporter introduced in PyTorch 2.1+.

        Captures the full computation graph via ``torch.export`` before
        lowering to ONNX, giving better graph coverage and fewer tracing
        surprises compared to the legacy TorchScript path.
        """
        export_output = torch.onnx.export(
            self.pytorch_model,
            (dummy_input,),
            dynamo=True,
        )
        export_output.save(output_path)

    def get_dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """Return a dummy input tensor with the given batch size."""
        return self._make_dummy_input(batch_size)
