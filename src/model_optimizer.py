"""
Model Optimizer Module
Handles ONNX model optimization including graph simplification and quantization.

Optimization techniques provided:
  1. Graph simplification  – onnx-simplifier (folds constants, removes dead nodes)
  2. Dynamic quantization  – INT8 weights, activations quantized at runtime (CPU)
  3. FP16 conversion       – halves model size, ideal for GPU inference
  4. ORT graph optimization – SessionOptions.ORT_ENABLE_ALL applied when loading
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize ONNX models using various compression and precision techniques."""

    def __init__(self, onnx_model_path: str):
        """
        Initialize the optimizer.

        Args:
            onnx_model_path: Path to the source ONNX model.
        """
        self.onnx_model_path = onnx_model_path
        self._model_dir = Path(onnx_model_path).parent
        self._stem = Path(onnx_model_path).stem

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_output(self, suffix: str) -> str:
        return str(self._model_dir / f"{self._stem}{suffix}.onnx")

    @staticmethod
    def _make_session_options(opt_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL) -> ort.SessionOptions:
        """
        Build an ORT SessionOptions with a chosen graph optimization level.

        ORT_ENABLE_ALL (level 99) activates:
          - Basic optimisations (node fusions, constant folding)
          - Extended optimisations (layout optimisation, memory reuse)
          - Execution-provider–specific optimisations
        """
        opts = ort.SessionOptions()
        opts.graph_optimization_level = opt_level
        return opts

    # ------------------------------------------------------------------
    # Optimization methods
    # ------------------------------------------------------------------

    def simplify(self, output_path: Optional[str] = None) -> str:
        """
        Simplify ONNX graph structure using onnx-simplifier.

        Removes redundant nodes, folds constants, and canonicalises the graph
        to help downstream tools (quantizers, ORT) work more efficiently.

        Args:
            output_path: Destination path; defaults to ``<stem>_simplified.onnx``.

        Returns:
            Path to the simplified model.
        """
        output_path = output_path or self._default_output("_simplified")
        model = onnx.load(self.onnx_model_path)
        model_simp, check = simplify(model)
        if not check:
            logger.warning("onnx-simplifier validation failed; saving output anyway.")
        onnx.save(model_simp, output_path)
        logger.info("Graph simplified → %s", output_path)
        return output_path

    def dynamic_quantize(self, output_path: Optional[str] = None) -> str:
        """
        Apply dynamic INT8 quantization (weights only, CPU-friendly).

        Weight tensors are pre-quantized to QUInt8; activations are quantized
        on-the-fly during inference.  Typically yields ~4× size reduction with
        minimal accuracy loss on NLP/vision models.

        Args:
            output_path: Destination path; defaults to ``<stem>_dynamic_quant.onnx``.

        Returns:
            Path to the quantized model.
        """
        output_path = output_path or self._default_output("_dynamic_quant")
        quantize_dynamic(
            self.onnx_model_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
        logger.info("Dynamic INT8 quantization → %s", output_path)
        return output_path

    def fp16_quantize(self, output_path: Optional[str] = None) -> str:
        """
        Convert model weights and activations to float16.

        Halves model file size and memory footprint; best suited for CUDA
        devices that have native FP16 tensor-core support.

        Args:
            output_path: Destination path; defaults to ``<stem>_fp16.onnx``.

        Returns:
            Path to the FP16 model.

        Raises:
            ImportError: If ``onnxconverter-common`` is not installed.
        """
        try:
            from onnxconverter_common import float16
        except ImportError as exc:
            raise ImportError(
                "FP16 conversion requires 'onnxconverter-common'. "
                "Install it with: pip install onnxconverter-common"
            ) from exc

        output_path = output_path or self._default_output("_fp16")
        model = onnx.load(self.onnx_model_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, output_path)
        logger.info("FP16 conversion → %s", output_path)
        return output_path

    def create_optimized_session(
        self,
        model_path: Optional[str] = None,
        providers: Optional[list[str]] = None,
    ) -> ort.InferenceSession:
        """
        Create an ORT InferenceSession with full graph optimization enabled.

        Using ``ORT_ENABLE_ALL`` lets ORT apply operator fusions (e.g. LayerNorm,
        Attention), memory layout transformations, and EP-specific kernels at
        session-creation time—no extra file needed.

        Args:
            model_path: Path to ONNX model; defaults to the source model.
            providers: ORT execution providers; defaults to CPU only.

        Returns:
            An optimized ``ort.InferenceSession``.
        """
        path = model_path or self.onnx_model_path
        providers = providers or ["CPUExecutionProvider"]
        opts = self._make_session_options()
        session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        logger.info(
            "ORT session created with ORT_ENABLE_ALL optimization (model: %s)", path
        )
        return session

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_model_size(model_path: str) -> float:
        """Return model file size in megabytes."""
        return os.path.getsize(model_path) / (1024 * 1024)
