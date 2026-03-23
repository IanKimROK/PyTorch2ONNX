"""Tests for src.benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.benchmark import Benchmark, _cuda_sync


# ---------------------------------------------------------------------------
# _cuda_sync helper
# ---------------------------------------------------------------------------

class TestCudaSync:
    def test_no_error_when_disabled(self):
        _cuda_sync(enabled=False)  # should be a no-op

    def test_no_error_on_cpu_even_when_enabled(self):
        # torch.cuda.synchronize() on a CPU-only machine raises no error
        # when CUDA is not available; this just ensures the call path is safe.
        if not torch.cuda.is_available():
            _cuda_sync(enabled=True)


# ---------------------------------------------------------------------------
# Benchmark initialisation
# ---------------------------------------------------------------------------

class TestBenchmarkInit:
    def test_default_device_is_cpu(self):
        b = Benchmark()
        assert b.device == "cpu"

    def test_providers_cpu(self):
        b = Benchmark(device="cpu")
        assert b.providers == ["CPUExecutionProvider"]

    def test_num_warmup_stored(self):
        b = Benchmark(num_warmup=5)
        assert b.num_warmup == 5

    def test_cuda_flag_false_on_cpu(self):
        b = Benchmark(device="cpu")
        assert b._use_cuda is False


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def test_identical_arrays_give_zero_mse(self):
        arr = np.random.rand(1, 10).astype(np.float32)
        metrics = Benchmark.calculate_metrics(arr, arr.copy())
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-9)

    def test_identical_arrays_give_unit_cosine(self):
        arr = np.random.rand(1, 10).astype(np.float32)
        metrics = Benchmark.calculate_metrics(arr, arr.copy())
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_arrays_give_zero_cosine(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        metrics = Benchmark.calculate_metrics(a, b)
        assert metrics["cosine_similarity"] == pytest.approx(0.0, abs=1e-6)

    def test_returns_required_keys(self):
        a = np.ones((1, 5), dtype=np.float32)
        b = np.zeros((1, 5), dtype=np.float32)
        metrics = Benchmark.calculate_metrics(a, b)
        assert "mse" in metrics
        assert "cosine_similarity" in metrics

    def test_mse_scales_correctly(self):
        a = np.array([[2.0, 2.0]])
        b = np.array([[0.0, 0.0]])
        metrics = Benchmark.calculate_metrics(a, b)
        assert metrics["mse"] == pytest.approx(4.0, rel=1e-5)


# ---------------------------------------------------------------------------
# measure_pytorch_inference
# ---------------------------------------------------------------------------

class TestMeasurePytorchInference:
    def test_returns_tuple_of_float_and_ndarray(self, tiny_model, dummy_input):
        b = Benchmark(device="cpu")
        time_ms, output = b.measure_pytorch_inference(tiny_model, dummy_input, num_runs=3)
        assert isinstance(time_ms, float)
        assert isinstance(output, np.ndarray)

    def test_latency_is_positive(self, tiny_model, dummy_input):
        b = Benchmark(device="cpu")
        time_ms, _ = b.measure_pytorch_inference(tiny_model, dummy_input, num_runs=3)
        assert time_ms > 0

    def test_output_shape(self, tiny_model, input_shape):
        b = Benchmark(device="cpu")
        inp = torch.randn(*input_shape)
        _, output = b.measure_pytorch_inference(tiny_model, inp, num_runs=2)
        assert output.shape == (1, 10)


# ---------------------------------------------------------------------------
# measure_onnx_inference
# ---------------------------------------------------------------------------

class TestMeasureOnnxInference:
    def test_returns_tuple(self, tiny_onnx_path, dummy_array):
        b = Benchmark(device="cpu")
        time_ms, output = b.measure_onnx_inference(tiny_onnx_path, dummy_array, num_runs=3)
        assert isinstance(time_ms, float)
        assert isinstance(output, np.ndarray)

    def test_latency_positive(self, tiny_onnx_path, dummy_array):
        b = Benchmark(device="cpu")
        time_ms, _ = b.measure_onnx_inference(tiny_onnx_path, dummy_array, num_runs=3)
        assert time_ms > 0

    def test_output_shape(self, tiny_onnx_path, dummy_array):
        b = Benchmark(device="cpu")
        _, output = b.measure_onnx_inference(tiny_onnx_path, dummy_array, num_runs=2)
        assert output.shape == (1, 10)

    def test_pytorch_onnx_outputs_close(self, tiny_model, dummy_input, dummy_array, tiny_onnx_path):
        """PyTorch and ONNX outputs should agree to within float32 tolerance."""
        b = Benchmark(device="cpu")
        _, pt_out = b.measure_pytorch_inference(tiny_model, dummy_input, num_runs=1)
        _, onnx_out = b.measure_onnx_inference(tiny_onnx_path, dummy_array, num_runs=1)
        np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4, atol=1e-5)
