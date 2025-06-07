"""
Utility Module
Helper functions for image loading and preprocessing
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from typing import Tuple, Optional
import os


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
        
    def _get_transform(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def load_cifar10_sample(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load sample images from CIFAR-10 dataset.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            Tuple of (images tensor, labels tensor)
        """
        # Download CIFAR-10 if not exists
        dataset = datasets.CIFAR10(root='./data', train=False, 
                                  download=True, transform=self.transform)
        
        # Load samples
        images = []
        labels = []
        
        for i in range(min(num_samples, len(dataset))):
            img, label = dataset[i]
            images.append(img)
            labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)
    
    def create_dummy_batch(self, batch_size: int = 1, 
                          channels: int = 3,
                          height: int = 224,
                          width: int = 224) -> torch.Tensor:
        """
        Create dummy input batch for testing.
        
        Args:
            batch_size: Batch size
            channels: Number of channels
            height: Image height
            width: Image width
            
        Returns:
            Random tensor with specified shape
        """
        return torch.randn(batch_size, channels, height, width)


def check_gpu_availability():
    """Check and print GPU availability information."""
    print("\n🖥️  System Information:")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")
    

def create_output_directory(base_dir: str = "models"):
    """Create output directory for models."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir