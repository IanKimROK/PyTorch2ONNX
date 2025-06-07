"""
Model Optimizer Module
Handles ONNX model optimization including simplification and quantization
"""

import os
import onnx
from onnxsim import simplify
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from typing import Optional


class ModelOptimizer:
    """Optimize ONNX models using various techniques."""
    
    def __init__(self, onnx_model_path: str):
        """
        Initialize model optimizer.
        
        Args:
            onnx_model_path: Path to ONNX model
        """
        self.onnx_model_path = onnx_model_path
        self.model_dir = os.path.dirname(onnx_model_path)
        self.model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
        
    def simplify(self, output_path: Optional[str] = None) -> str:
        """
        Simplify ONNX model structure using onnx-simplifier.
        
        Args:
            output_path: Path to save simplified model
            
        Returns:
            Path to simplified model
        """
        if output_path is None:
            output_path = os.path.join(self.model_dir, f"{self.model_name}_simplified.onnx")
        
        # Load and simplify model
        model = onnx.load(self.onnx_model_path)
        model_simp, check = simplify(model)
        
        if not check:
            print("⚠️  Simplified model validation failed, but saving anyway...")
        
        # Save simplified model
        onnx.save(model_simp, output_path)
        print(f"✅ Model simplified and saved to: {output_path}")
        
        return output_path
    
    def dynamic_quantize(self, output_path: Optional[str] = None) -> str:
        """
        Apply dynamic quantization to ONNX model.
        
        Args:
            output_path: Path to save quantized model
            
        Returns:
            Path to quantized model
        """
        if output_path is None:
            output_path = os.path.join(self.model_dir, f"{self.model_name}_dynamic_quant.onnx")
        
        # Apply dynamic quantization
        quantize_dynamic(
            self.onnx_model_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
        
        print(f"✅ Dynamic quantization applied and saved to: {output_path}")
        return output_path
    
    def fp16_quantize(self, output_path: Optional[str] = None) -> str:
        """
        Convert model to float16 precision.
        
        Args:
            output_path: Path to save FP16 model
            
        Returns:
            Path to FP16 model
        """
        if output_path is None:
            output_path = os.path.join(self.model_dir, f"{self.model_name}_fp16.onnx")
        
        # Load model
        model = onnx.load(self.onnx_model_path)
        
        # Convert to FP16
        from onnxconverter_common import float16
        model_fp16 = float16.convert_float_to_float16(model)
        
        # Save FP16 model
        onnx.save(model_fp16, output_path)
        print(f"✅ FP16 conversion completed and saved to: {output_path}")
        
        return output_path
    
    def get_model_size(self, model_path: str) -> float:
        """
        Get model file size in MB.
        
        Args:
            model_path: Path to model file
            
        Returns:
            File size in MB
        """
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb