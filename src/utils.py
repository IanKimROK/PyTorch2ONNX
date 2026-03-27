"""
Utility Module
Helper functions for image loading, preprocessing, and system checks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class ImageLoader:
    """Load and preprocess images for model inference."""

    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image loader.

        Args:
            input_size: Target image size (height, width)
        """
        self.input_size = input_size
        self.transform = self._get_transform()

    def _get_transform(self) -> transforms.Compose:
        """Build ImageNet-standard image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor of shape (1, C, H, W)
        """
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def load_cifar10_sample(
        self, num_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load sample images from CIFAR-10 dataset.

        Args:
            num_samples: Number of samples to load

        Returns:
            Tuple of (images tensor, labels tensor)
        """
        dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )
        num_samples = min(num_samples, len(dataset))
        images, labels = zip(*(dataset[i] for i in range(num_samples)))
        return torch.stack(list(images)), torch.tensor(list(labels))

    def create_dummy_batch(
        self,
        batch_size: int = 1,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
    ) -> torch.Tensor:
        """
        Create a random input batch for testing.

        Args:
            batch_size: Batch size
            channels: Number of channels
            height: Image height
            width: Image width

        Returns:
            Random tensor of shape (batch_size, channels, height, width)
        """
        return torch.randn(batch_size, channels, height, width)


def check_gpu_availability() -> None:
    """Log GPU / ONNX Runtime availability information."""
    logger.info("=== System Information ===")
    cuda_available = torch.cuda.is_available()
    logger.info("PyTorch CUDA available: %s", cuda_available)
    if cuda_available:
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA version: %s", torch.version.cuda)
    logger.info("ONNX Runtime providers: %s", ort.get_available_providers())


def create_output_directory(base_dir: str = "models") -> str:
    """Create output directory for models and return its path."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return base_dir
