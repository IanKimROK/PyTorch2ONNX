"""
Model Converter Module
Handles PyTorch to ONNX conversion with dynamic batch size support
"""

import torch
import torch.onnx
import torchvision.models as models
from typing import Tuple, Optional
import os


class ModelConverter:
    """Convert PyTorch models to ONNX format with dynamic batch size support."""
    
    def __init__(self, model_name: str = 'resnet18', 
                 input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                 pretrained: bool = True):
        """
        Initialize model converter.
        
        Args:
            model_name: Name of the model (from torchvision.models)
            input_shape: Input tensor shape (batch, channels, height, width)
            pretrained: Whether to use pretrained weights
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.pretrained = pretrained
        
        # Load PyTorch model
        self.pytorch_model = self._load_pytorch_model()
        self.pytorch_model.eval()
        
    def _load_pytorch_model(self):
        """Load PyTorch model from torchvision."""
        if hasattr(models, self.model_name):
            model_func = getattr(models, self.model_name)
            if self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                model = model_func(weights='IMAGENET1K_V1' if self.pretrained else None)
            else:
                model = model_func(pretrained=self.pretrained)
            return model
        else:
            raise ValueError(f"Model {self.model_name} not found in torchvision.models")
    
    def convert_to_onnx(self, output_path: str, 
                       input_names: Optional[list] = None,
                       output_names: Optional[list] = None,
                       opset_version: int = 11) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Default names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # Create dummy input
        dummy_input = torch.randn(*self.input_shape)
        
        # Dynamic axes for batch size
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
        
        # Export to ONNX
        torch.onnx.export(
            self.pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"✅ Successfully converted {self.model_name} to ONNX: {output_path}")
        return output_path
    
    def get_dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """Get dummy input tensor with specified batch size."""
        shape = list(self.input_shape)
        shape[0] = batch_size
        return torch.randn(*shape)