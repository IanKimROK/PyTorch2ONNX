"""Tests for src.utils."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.utils import ImageLoader, check_gpu_availability, create_output_directory


# ---------------------------------------------------------------------------
# ImageLoader
# ---------------------------------------------------------------------------

class TestImageLoader:
    def test_default_input_size(self):
        loader = ImageLoader()
        assert loader.input_size == (224, 224)

    def test_custom_input_size(self):
        loader = ImageLoader(input_size=(128, 128))
        assert loader.input_size == (128, 128)

    def test_create_dummy_batch_default_shape(self):
        loader = ImageLoader()
        batch = loader.create_dummy_batch()
        assert batch.shape == (1, 3, 224, 224)

    def test_create_dummy_batch_custom_shape(self):
        loader = ImageLoader()
        batch = loader.create_dummy_batch(batch_size=4, channels=1, height=64, width=64)
        assert batch.shape == (4, 1, 64, 64)

    def test_create_dummy_batch_returns_tensor(self):
        loader = ImageLoader()
        batch = loader.create_dummy_batch()
        assert isinstance(batch, torch.Tensor)

    def test_create_dummy_batch_dtype_float32(self):
        loader = ImageLoader()
        batch = loader.create_dummy_batch()
        assert batch.dtype == torch.float32

    def test_transform_pipeline_is_compose(self):
        from torchvision import transforms
        loader = ImageLoader()
        assert isinstance(loader.transform, transforms.Compose)

    def test_load_image_from_file(self, tmp_path):
        """Save a small RGB image and check that load_image returns the right shape."""
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (256, 256), color=(128, 64, 32))
        img_path = str(tmp_path / "test.jpg")
        img.save(img_path)

        loader = ImageLoader(input_size=(224, 224))
        tensor = loader.load_image(img_path)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(tensor, torch.Tensor)


# ---------------------------------------------------------------------------
# create_output_directory
# ---------------------------------------------------------------------------

class TestCreateOutputDirectory:
    def test_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "output_models")
        result = create_output_directory(new_dir)
        assert os.path.isdir(result)

    def test_returns_path_string(self, tmp_path):
        new_dir = str(tmp_path / "ret_test")
        result = create_output_directory(new_dir)
        assert result == new_dir

    def test_idempotent_on_existing_dir(self, tmp_path):
        existing = str(tmp_path / "existing")
        create_output_directory(existing)
        create_output_directory(existing)  # should not raise
        assert os.path.isdir(existing)

    def test_creates_nested_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        create_output_directory(nested)
        assert os.path.isdir(nested)


# ---------------------------------------------------------------------------
# check_gpu_availability
# ---------------------------------------------------------------------------

class TestCheckGpuAvailability:
    def test_runs_without_error(self):
        """check_gpu_availability should not raise regardless of GPU presence."""
        check_gpu_availability()

    def test_logs_cuda_status(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="src.utils"):
            check_gpu_availability()
        assert any("CUDA" in msg or "PyTorch" in msg for msg in caplog.messages)
