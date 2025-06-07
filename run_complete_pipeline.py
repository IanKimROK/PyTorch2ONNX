#!/usr/bin/env python3
"""
Complete PyTorch to ONNX Pipeline Script
This is a standalone script that includes all functionality in one file
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

# Core imports
import torch
import torch.onnx
import torchvision.models as models
from torchvision import transforms, datasets
from PIL import Image

# ONNX imports
import onnx
from onnxsim import simplify
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Utilities
from tabulate import tabulate
from scipy.spatial.distance import cosine


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    os.system("pip install torch torchvision onnx onnxruntime onnx-simplifier numpy pillow tabulate scipy")
    
    # Optional for GPU support
    print("\n💡 For GPU support, also install:")
    print("   pip install onnxruntime-gpu")
    
    # Optional for FP16 conversion
    print("\n💡 For FP16 conversion support, also install:")
    print("   pip install onnxconverter-common")


class CompletePipeline:
    """Complete pipeline for PyTorch to ONNX conversion and optimization."""
    
    def __init__(self, model_name='resnet18', input_size=224, device='cpu'):
        self.model_name = model_name
        self.input_shape = (1, 3, input_size, input_size)
        self.device = device
        self.output_dir = 'models'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load PyTorch model
        self.pytorch_model = self._load_pytorch_model()
        
    def _load_pytorch_model(self):
        """Load PyTorch model from torchvision."""
        print(f"📥 Loading {self.model_name} model...")
        
        if hasattr(models, self.model_name):
            model_func = getattr(models, self.model_name)
            if self.model_name.startswith('resnet'):
                model = model_func(weights='IMAGENET1K_V1')
            else:
                model = model_func(pretrained=True)
            model.eval()
            return model
        else:
            raise ValueError(f"Model {self.model_name} not found")
    
    def convert_to_onnx(self):
        """Convert PyTorch model to ONNX."""
        print("\n🔄 Converting to ONNX...")
        
        onnx_path = os.path.join(self.output_dir, f"{self.model_name}.onnx")
        dummy_input = torch.randn(*self.input_shape)
        
        # Dynamic axes for batch size
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        torch.onnx.export(
            self.pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"✅ ONNX model saved to: {onnx_path}")
        return onnx_path
    
    def optimize_models(self, onnx_path):
        """Apply various optimizations to ONNX model."""
        optimized_models = {}
        
        # 1. Simplify
        print("\n🔧 Applying onnx-simplifier...")
        simplified_path = os.path.join(self.output_dir, f"{self.model_name}_simplified.onnx")
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        onnx.save(model_simp, simplified_path)
        optimized_models['simplified'] = simplified_path
        print(f"✅ Simplified model saved to: {simplified_path}")
        
        # 2. Dynamic Quantization
        print("\n🔧 Applying dynamic quantization...")
        dynamic_quant_path = os.path.join(self.output_dir, f"{self.model_name}_dynamic_quant.onnx")
        quantize_dynamic(onnx_path, dynamic_quant_path, weight_type=QuantType.QUInt8)
        optimized_models['dynamic_quant'] = dynamic_quant_path
        print(f"✅ Dynamic quantized model saved to: {dynamic_quant_path}")
        
        # 3. FP16 Quantization
        try:
            print("\n🔧 Applying FP16 conversion...")
            from onnxconverter_common import float16
            fp16_path = os.path.join(self.output_dir, f"{self.model_name}_fp16.onnx")
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, fp16_path)
            optimized_models['fp16'] = fp16_path
            print(f"✅ FP16 model saved to: {fp16_path}")
        except ImportError:
            print("⚠️  onnxconverter-common not installed. Skipping FP16 conversion.")
        
        return optimized_models
    
    def benchmark_models(self, onnx_path, optimized_models, num_runs=100):
        """Benchmark all models."""
        print("\n📊 Benchmarking models...")
        
        results = []
        dummy_input = torch.randn(*self.input_shape)
        
        # Get available providers
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 1. Benchmark PyTorch
        print("⏱️  Benchmarking PyTorch model...")
        pt_time, pt_output = self._benchmark_pytorch(dummy_input, num_runs)
        pt_size = sum(p.numel() * p.element_size() for p in self.pytorch_model.parameters()) / (1024**2)
        
        results.append({
            'Model': 'PyTorch (Original)',
            'File Size (MB)': f'{pt_size:.1f}',
            'Inference Time (ms)': f'{pt_time:.2f}',
            'MSE': '0.0000',
            'Cosine Similarity': '1.0000'
        })
        
        # 2. Benchmark ONNX models
        all_models = {'ONNX': onnx_path}
        all_models.update({
            'ONNX Simplified': optimized_models.get('simplified'),
            'ONNX Dynamic Quant': optimized_models.get('dynamic_quant'),
            'ONNX FP16': optimized_models.get('fp16')
        })
        
        for name, path in all_models.items():
            if path and os.path.exists(path):
                print(f"⏱️  Benchmarking {name}...")
                onnx_time, onnx_output = self._benchmark_onnx(path, dummy_input.numpy(), providers, num_runs)
                file_size = os.path.getsize(path) / (1024**2)
                
                # Calculate accuracy metrics
                mse, cos_sim = self._calculate_metrics(pt_output, onnx_output)
                
                results.append({
                    'Model': name,
                    'File Size (MB)': f'{file_size:.1f}',
                    'Inference Time (ms)': f'{onnx_time:.2f}',
                    'MSE': f'{mse:.2e}',
                    'Cosine Similarity': f'{cos_sim:.4f}'
                })
        
        # Print results
        print("\n📈 Model Performance Comparison")
        print(tabulate(results, headers='keys', tablefmt='grid'))
        
        # GPU vs CPU comparison
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            self._compare_gpu_cpu(onnx_path, dummy_input.numpy())
        
        return results
    
    def _benchmark_pytorch(self, input_tensor, num_runs):
        """Benchmark PyTorch model."""
        model = self.pytorch_model
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()
        
        # Warmup
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Measure
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
                torch.cuda.synchronize() if self.device == 'cuda' else None
        
        elapsed = (time.time() - start) / num_runs * 1000
        return elapsed, output.cpu().numpy()
    
    def _benchmark_onnx(self, model_path, input_array, providers, num_runs):
        """Benchmark ONNX model."""
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        # Warmup
        _ = session.run(None, {input_name: input_array})
        
        # Measure
        start = time.time()
        for _ in range(num_runs):
            output = session.run(None, {input_name: input_array})
        
        elapsed = (time.time() - start) / num_runs * 1000
        return elapsed, output[0]
    
    def _calculate_metrics(self, output1, output2):
        """Calculate MSE and cosine similarity."""
        flat1 = output1.flatten()
        flat2 = output2.flatten()
        
        mse = np.mean((flat1 - flat2) ** 2)
        cos_sim = 1 - cosine(flat1, flat2)
        
        return mse, cos_sim
    
    def _compare_gpu_cpu(self, onnx_path, input_array):
        """Compare GPU vs CPU performance."""
        print("\n🖥️  GPU vs CPU Comparison")
        
        # CPU
        cpu_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = cpu_session.get_inputs()[0].name
        
        start = time.time()
        for _ in range(100):
            _ = cpu_session.run(None, {input_name: input_array})
        cpu_time = (time.time() - start) / 100 * 1000
        
        # GPU
        gpu_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        
        start = time.time()
        for _ in range(100):
            _ = gpu_session.run(None, {input_name: input_array})
        gpu_time = (time.time() - start) / 100 * 1000
        
        print(f"CPU Inference: {cpu_time:.1f} ms")
        print(f"GPU Inference: {gpu_time:.1f} ms")
        print(f"GPU Speedup: {cpu_time/gpu_time:.2f}x")
    
    def test_dynamic_batch(self, onnx_path):
        """Test dynamic batch sizes."""
        print("\n📦 Testing dynamic batch sizes...")
        
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        for batch_size in [1, 4, 8, 16]:
            try:
                test_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
                _ = session.run(None, {input_name: test_input})
                print(f"✅ Batch size {batch_size}: OK")
            except Exception as e:
                print(f"❌ Batch size {batch_size}: Failed - {str(e)}")
    
    def test_real_images(self, onnx_path):
        """Test with real CIFAR-10 images."""
        print("\n🖼️  Testing with real images (CIFAR-10)...")
        
        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Test a few images
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        for i in range(5):
            img, label = dataset[i]
            img_batch = img.unsqueeze(0).numpy()
            output = session.run(None, {input_name: img_batch})
            predicted = np.argmax(output[0])
            print(f"Image {i}: True label={label}, Predicted={predicted}")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline."""
        print("🚀 Starting PyTorch to ONNX Conversion Pipeline")
        print("=" * 60)
        
        # System info
        print(f"\n💻 System Information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available providers: {ort.get_available_providers()}")
        
        # Step 1: Convert to ONNX
        onnx_path = self.convert_to_onnx()
        
        # Step 2: Optimize models
        optimized_models = self.optimize_models(onnx_path)
        
        # Step 3: Benchmark
        results = self.benchmark_models(onnx_path, optimized_models)
        
        # Step 4: Test dynamic batching
        self.test_dynamic_batch(onnx_path)
        
        # Step 5: Test with real images
        self.test_real_images(onnx_path)
        
        print("\n✨ Pipeline completed successfully!")
        print(f"📁 All models saved in: {self.output_dir}/")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Complete PyTorch to ONNX Pipeline")
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--input-size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
        return
    
    # Run pipeline
    pipeline = CompletePipeline(
        model_name=args.model,
        input_size=args.input_size,
        device=args.device
    )
    
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()