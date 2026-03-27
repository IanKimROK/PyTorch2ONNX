"""
Benchmark Module
Measures inference latency, file size, and output accuracy for all model variants.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from scipy.spatial.distance import cosine
from tabulate import tabulate

logger = logging.getLogger(__name__)


def _cuda_sync(enabled: bool) -> None:
    """Synchronize CUDA device when running on GPU."""
    if enabled:
        torch.cuda.synchronize()


class Benchmark:
    """Benchmark inference latency, file size, and numerical accuracy."""

    def __init__(self, device: str = "cpu", num_warmup: int = 10):
        """
        Initialize the benchmark runner.

        Args:
            device: ``'cpu'`` or ``'cuda'``.
            num_warmup: Number of warmup iterations before timing starts.
                        More warmup reduces JIT / kernel-launch variance.
        """
        self.device = device
        self.num_warmup = num_warmup
        self.providers = self._get_providers()
        self._use_cuda = device == "cuda" and torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_providers(self) -> List[str]:
        """Return the best available ORT execution providers for this device."""
        if (
            self.device == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _make_session_options() -> ort.SessionOptions:
        """ORT session with maximum graph optimization."""
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    # ------------------------------------------------------------------
    # PyTorch inference
    # ------------------------------------------------------------------

    def measure_pytorch_inference(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
    ) -> Tuple[float, np.ndarray]:
        """
        Measure PyTorch model inference latency.

        Args:
            model: PyTorch model (will be moved to ``self.device``).
            input_tensor: Input tensor (batch dimension should match benchmark intent).
            num_runs: Number of timed iterations.

        Returns:
            ``(avg_latency_ms, output_numpy)``
        """
        model.eval()
        if self._use_cuda:
            model = model.cuda()
            input_tensor = input_tensor.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model(input_tensor)
        _cuda_sync(self._use_cuda)

        # Timed runs
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
                _cuda_sync(self._use_cuda)
        elapsed_ms = (time.perf_counter() - start) / num_runs * 1000

        return elapsed_ms, output.cpu().numpy()

    # ------------------------------------------------------------------
    # ONNX Runtime inference
    # ------------------------------------------------------------------

    def measure_onnx_inference(
        self,
        onnx_path: str,
        input_array: np.ndarray,
        num_runs: int = 100,
        providers: Optional[List[str]] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Measure ONNX Runtime inference latency.

        Args:
            onnx_path: Path to the ``.onnx`` model file.
            input_array: Input array (must match model's expected dtype/shape).
            num_runs: Number of timed iterations.
            providers: ORT providers; defaults to ``self.providers``.

        Returns:
            ``(avg_latency_ms, output_numpy)``
        """
        providers = providers or self.providers
        session = ort.InferenceSession(
            onnx_path,
            sess_options=self._make_session_options(),
            providers=providers,
        )
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(self.num_warmup):
            session.run(None, {input_name: input_array})

        # Timed runs
        start = time.perf_counter()
        for _ in range(num_runs):
            output = session.run(None, {input_name: input_array})
        elapsed_ms = (time.perf_counter() - start) / num_runs * 1000

        return elapsed_ms, output[0]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(
        output1: np.ndarray, output2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute numerical similarity between two model outputs.

        Args:
            output1: Reference output (e.g. PyTorch FP32).
            output2: Candidate output (e.g. quantized ONNX).

        Returns:
            Dict with ``mse`` and ``cosine_similarity`` keys.
        """
        flat1 = output1.flatten().astype(np.float64)
        flat2 = output2.flatten().astype(np.float64)
        mse = float(np.mean((flat1 - flat2) ** 2))
        cos_sim = float(1.0 - cosine(flat1, flat2))
        return {"mse": mse, "cosine_similarity": cos_sim}

    # ------------------------------------------------------------------
    # Full comparison
    # ------------------------------------------------------------------

    def compare_all_models(
        self,
        pytorch_model: torch.nn.Module,
        onnx_path: str,
        optimized_paths: List[str],
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        num_runs: int = 100,
    ) -> List[Dict]:
        """
        Benchmark PyTorch model and all ONNX variants side-by-side.

        Args:
            pytorch_model: Original PyTorch model.
            onnx_path: Path to the base ONNX model.
            optimized_paths: Paths to simplified / quantized / FP16 variants.
            input_shape: Tensor shape used for all inference runs.
            num_runs: Number of timed iterations per model.

        Returns:
            List of result dicts (one per model variant).
        """
        input_tensor = torch.randn(*input_shape)
        input_array = input_tensor.numpy()

        results: List[Dict] = []

        # --- PyTorch baseline ---
        logger.info("Benchmarking PyTorch baseline …")
        pt_time, pt_output = self.measure_pytorch_inference(
            pytorch_model, input_tensor, num_runs
        )
        pt_size_mb = (
            sum(p.numel() * p.element_size() for p in pytorch_model.parameters())
            / (1024 * 1024)
        )
        results.append(
            {
                "Model": "PyTorch (Original)",
                "File Size (MB)": f"{pt_size_mb:.1f}",
                "Inference Time (ms)": f"{pt_time:.2f}",
                "MSE": "0.000e+00",
                "Cosine Similarity": "1.0000",
            }
        )

        # --- ONNX variants ---
        model_labels = [
            "ONNX",
            "ONNX Simplified",
            "ONNX Dynamic Quant (INT8)",
            "ONNX FP16",
        ]
        all_onnx_paths = [onnx_path, *optimized_paths]

        for path, label in zip(all_onnx_paths, model_labels):
            if not os.path.exists(path):
                logger.warning("Model file not found, skipping: %s", path)
                continue

            logger.info("Benchmarking %s …", label)
            file_size_mb = os.path.getsize(path) / (1024 * 1024)

            try:
                onnx_time, onnx_output = self.measure_onnx_inference(
                    path, input_array, num_runs
                )
                metrics = self.calculate_metrics(pt_output, onnx_output)
                results.append(
                    {
                        "Model": label,
                        "File Size (MB)": f"{file_size_mb:.1f}",
                        "Inference Time (ms)": f"{onnx_time:.2f}",
                        "MSE": f"{metrics['mse']:.3e}",
                        "Cosine Similarity": f"{metrics['cosine_similarity']:.4f}",
                    }
                )
            except Exception:
                logger.exception("Failed to benchmark %s", label)

        # --- Print table ---
        print("\n=== Model Performance Comparison ===")
        print(tabulate(results, headers="keys", tablefmt="grid"))

        # --- GPU vs CPU breakdown ---
        if self._use_cuda:
            self._compare_gpu_cpu(onnx_path, input_array, num_runs)

        return results

    def _compare_gpu_cpu(
        self, onnx_path: str, input_array: np.ndarray, num_runs: int = 100
    ) -> None:
        """Log GPU vs CPU throughput comparison for the base ONNX model."""
        cpu_time, _ = self.measure_onnx_inference(
            onnx_path, input_array, num_runs, providers=["CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.info("CUDAExecutionProvider not available; skipping GPU comparison.")
            return

        gpu_time, _ = self.measure_onnx_inference(
            onnx_path,
            input_array,
            num_runs,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        logger.info(
            "GPU vs CPU — CPU: %.2f ms | GPU: %.2f ms | Speedup: %.2fx",
            cpu_time,
            gpu_time,
            speedup,
        )
        print(f"\n=== GPU vs CPU ===")
        print(f"CPU : {cpu_time:.2f} ms")
        print(f"GPU : {gpu_time:.2f} ms")
        print(f"Speedup: {speedup:.2f}×")
