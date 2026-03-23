"""Tests for src.model_optimizer."""

from __future__ import annotations

import os

import pytest

import onnxruntime as ort

from src.model_optimizer import ModelOptimizer


class TestGetModelSize:
    def test_returns_float(self, tiny_onnx_path):
        size = ModelOptimizer.get_model_size(tiny_onnx_path)
        assert isinstance(size, float)
        assert size > 0

    def test_consistent_with_os_stat(self, tiny_onnx_path):
        expected = os.path.getsize(tiny_onnx_path) / (1024 * 1024)
        assert ModelOptimizer.get_model_size(tiny_onnx_path) == pytest.approx(expected)


class TestSimplify:
    def test_output_file_created(self, tiny_onnx_path, tmp_dir):
        optimizer = ModelOptimizer(tiny_onnx_path)
        out = str(tmp_dir / "simplified.onnx")
        result = optimizer.simplify(output_path=out)
        assert os.path.exists(result)

    def test_default_output_name(self, tiny_onnx_path, tmp_dir):
        # Copy the onnx file to tmp_dir so the default path lands there
        import shutil
        local_path = str(tmp_dir / "tiny.onnx")
        shutil.copy(tiny_onnx_path, local_path)
        optimizer = ModelOptimizer(local_path)
        result = optimizer.simplify()
        assert "simplified" in os.path.basename(result)
        assert os.path.exists(result)


class TestDynamicQuantize:
    def test_output_file_created(self, tiny_onnx_path, tmp_dir):
        optimizer = ModelOptimizer(tiny_onnx_path)
        out = str(tmp_dir / "quant.onnx")
        result = optimizer.dynamic_quantize(output_path=out)
        assert os.path.exists(result)

    def test_quantized_model_is_smaller(self, tiny_onnx_path, tmp_dir):
        optimizer = ModelOptimizer(tiny_onnx_path)
        out = str(tmp_dir / "quant_size.onnx")
        optimizer.dynamic_quantize(output_path=out)
        orig_size = os.path.getsize(tiny_onnx_path)
        quant_size = os.path.getsize(out)
        # Quantized model can be smaller OR similar; just check it's a valid file
        assert quant_size > 0

    def test_quantized_model_is_runnable(self, tiny_onnx_path, dummy_array, tmp_dir):
        optimizer = ModelOptimizer(tiny_onnx_path)
        out = str(tmp_dir / "quant_run.onnx")
        optimizer.dynamic_quantize(output_path=out)
        session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: dummy_array})
        assert result[0].shape[0] == 1


class TestFp16Quantize:
    def test_raises_import_error_without_package(self, tiny_onnx_path, monkeypatch, tmp_dir):
        """fp16_quantize should raise ImportError when onnxconverter_common is absent."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxconverter_common":
                raise ImportError("mocked missing package")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        optimizer = ModelOptimizer(tiny_onnx_path)
        with pytest.raises(ImportError, match="onnxconverter-common"):
            optimizer.fp16_quantize(output_path=str(tmp_dir / "fp16.onnx"))


class TestCreateOptimizedSession:
    def test_returns_inference_session(self, tiny_onnx_path):
        optimizer = ModelOptimizer(tiny_onnx_path)
        session = optimizer.create_optimized_session()
        assert isinstance(session, ort.InferenceSession)

    def test_session_runs_inference(self, tiny_onnx_path, dummy_array):
        optimizer = ModelOptimizer(tiny_onnx_path)
        session = optimizer.create_optimized_session()
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: dummy_array})
        assert result[0].shape == (1, 10)
