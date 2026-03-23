"""Tests for src.model_converter."""

from __future__ import annotations

import os

import pytest
import torch

from src.model_converter import ConversionConfig, ModelConverter


# ---------------------------------------------------------------------------
# ConversionConfig
# ---------------------------------------------------------------------------

class TestConversionConfig:
    def test_defaults(self):
        cfg = ConversionConfig()
        assert cfg.model_name == "resnet18"
        assert cfg.input_shape == (1, 3, 224, 224)
        assert cfg.pretrained is True
        assert cfg.opset_version == 17
        assert cfg.use_dynamo is False
        assert cfg.input_names == ["input"]
        assert cfg.output_names == ["output"]

    def test_custom_values(self):
        cfg = ConversionConfig(
            model_name="mobilenet_v3_small",
            input_shape=(1, 3, 128, 128),
            pretrained=False,
            opset_version=13,
        )
        assert cfg.model_name == "mobilenet_v3_small"
        assert cfg.input_shape == (1, 3, 128, 128)
        assert cfg.pretrained is False
        assert cfg.opset_version == 13

    def test_input_names_default_factory(self):
        """Each instance should get its own list (not a shared default)."""
        cfg1 = ConversionConfig()
        cfg2 = ConversionConfig()
        cfg1.input_names.append("extra")
        assert "extra" not in cfg2.input_names


# ---------------------------------------------------------------------------
# ModelConverter – model loading
# ---------------------------------------------------------------------------

class TestModelConverterLoad:
    def test_invalid_model_raises(self):
        cfg = ConversionConfig(model_name="not_a_real_model_xyz", pretrained=False)
        with pytest.raises(ValueError, match="not found"):
            ModelConverter(config=cfg)

    def test_load_without_pretrained(self):
        cfg = ConversionConfig(model_name="resnet18", pretrained=False)
        conv = ModelConverter(config=cfg)
        assert isinstance(conv.pytorch_model, torch.nn.Module)

    def test_model_in_eval_mode(self):
        cfg = ConversionConfig(model_name="resnet18", pretrained=False)
        conv = ModelConverter(config=cfg)
        assert not conv.pytorch_model.training

    def test_backwards_compat_kwargs(self):
        """Old-style keyword arguments should still work."""
        conv = ModelConverter(model_name="resnet18", pretrained=False)
        assert conv.config.model_name == "resnet18"

    def test_dummy_input_shape(self):
        cfg = ConversionConfig(
            model_name="resnet18",
            input_shape=(2, 3, 64, 64),
            pretrained=False,
        )
        conv = ModelConverter(config=cfg)
        dummy = conv.get_dummy_input(batch_size=4)
        assert dummy.shape == (4, 3, 64, 64)


# ---------------------------------------------------------------------------
# ModelConverter – ONNX export
# ---------------------------------------------------------------------------

class TestConvertToOnnx:
    @pytest.fixture(scope="class")
    def converter(self):
        cfg = ConversionConfig(
            model_name="resnet18",
            input_shape=(1, 3, 64, 64),
            pretrained=False,
        )
        return ModelConverter(config=cfg)

    def test_output_file_created(self, converter, tmp_dir):
        out = str(tmp_dir / "model.onnx")
        path = converter.convert_to_onnx(output_path=out)
        assert os.path.exists(path)

    def test_returns_correct_path(self, converter, tmp_dir):
        out = str(tmp_dir / "model2.onnx")
        returned = converter.convert_to_onnx(output_path=out)
        assert returned == out

    def test_nested_directory_created(self, converter, tmp_dir):
        out = str(tmp_dir / "nested" / "deep" / "model.onnx")
        converter.convert_to_onnx(output_path=out)
        assert os.path.exists(out)

    def test_onnx_file_is_valid(self, converter, tmp_dir):
        import onnx
        out = str(tmp_dir / "valid.onnx")
        converter.convert_to_onnx(output_path=out)
        model = onnx.load(out)
        onnx.checker.check_model(model)
