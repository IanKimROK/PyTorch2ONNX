"""
Benchmark Module
Handles performance benchmarking and accuracy comparison
"""

import time
import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate
from scipy.spatial.distance import cosine
import os


class Benchmark:
    """Benchmark model performance including inference time and accuracy."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize benchmark.
        
        Args:
            device: Device to run benchmarks on ('cpu' or 'cuda')
        """
        self.device = device
        self.providers = self._get_providers()
        
    def _get_providers(self) -> List[str]:
        """Get available execution providers for ONNX Runtime."""
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    
    def measure_pytorch_inference(self, model: torch.nn.Module, 
                                input_tensor: torch.Tensor,
                                num_runs: int = 100) -> Tuple[float, np.ndarray]:
        """
        Measure PyTorch model inference time.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            num_runs: Number of inference runs for averaging
            
        Returns:
            Tuple of (average inference time in ms, output array)
        """
        model.eval()
        
        # Move to device if CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()
        
        # Warmup
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Measure inference time
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
                torch.cuda.synchronize() if self.device == 'cuda' else None
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        
        # Get output for comparison
        output_np = output.cpu().numpy()
        
        return avg_time_ms, output_np
    
    def measure_onnx_inference(self, onnx_path: str,
                             input_array: np.ndarray,
                             num_runs: int = 100) -> Tuple[float, np.ndarray]:
        """
        Measure ONNX model inference time.
        
        Args:
            onnx_path: Path to ONNX model
            input_array: Input array
            num_runs: Number of inference runs for averaging
            
        Returns:
            Tuple of (average inference time in ms, output array)
        """
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=self.providers)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Warmup
        _ = session.run(None, {input_name: input_array})
        
        # Measure inference time
        start_time = time.time()
        
        for _ in range(num_runs):
            output = session.run(None, {input_name: input_array})
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        
        return avg_time_ms, output[0]
    
    def calculate_metrics(self, output1: np.ndarray, output2: np.ndarray) -> Dict[str, float]:
        """
        Calculate comparison metrics between two outputs.
        
        Args:
            output1: First output array
            output2: Second output array
            
        Returns:
            Dictionary with MSE and cosine similarity
        """
        # Flatten arrays for comparison
        flat1 = output1.flatten()
        flat2 = output2.flatten()
        
        # Calculate MSE
        mse = np.mean((flat1 - flat2) ** 2)
        
        # Calculate cosine similarity
        cos_sim = 1 - cosine(flat1, flat2)
        
        return {
            'mse': mse,
            'cosine_similarity': cos_sim
        }
    
    def compare_all_models(self, pytorch_model: torch.nn.Module,
                         onnx_path: str,
                         optimized_paths: List[str],
                         input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                         num_runs: int = 100) -> Dict:
        """
        Compare all model variants.
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to base ONNX model
            optimized_paths: List of paths to optimized models
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with comparison results
        """
        results = []
        
        # Generate test input
        input_tensor = torch.randn(*input_shape)
        input_array = input_tensor.numpy()
        
        # Benchmark PyTorch model
        print("📊 Benchmarking PyTorch model...")
        pt_time, pt_output = self.measure_pytorch_inference(pytorch_model, input_tensor, num_runs)
        
        # Get PyTorch model size
        pt_size = sum(p.numel() * p.element_size() for p in pytorch_model.parameters()) / (1024 * 1024)
        
        results.append({
            'Model': 'PyTorch (Original)',
            'File Size (MB)': f'{pt_size:.1f}',
            'Inference Time (ms)': f'{pt_time:.1f}',
            'MSE': '0.0000',
            'Cosine Similarity': '1.0000'
        })
        
        # Benchmark ONNX models
        all_onnx_paths = [onnx_path] + optimized_paths
        model_names = ['ONNX', 'ONNX Simplified', 'ONNX Dynamic Quant', 'ONNX FP16']
        
        for path, name in zip(all_onnx_paths, model_names[:len(all_onnx_paths)]):
            if not os.path.exists(path):
                continue
                
            print(f"📊 Benchmarking {name}...")
            
            # Get file size
            file_size = os.path.getsize(path) / (1024 * 1024)
            
            # Measure inference
            onnx_time, onnx_output = self.measure_onnx_inference(path, input_array, num_runs)
            
            # Calculate metrics
            metrics = self.calculate_metrics(pt_output, onnx_output)
            
            results.append({
                'Model': name,
                'File Size (MB)': f'{file_size:.1f}',
                'Inference Time (ms)': f'{onnx_time:.1f}',
                'MSE': f'{metrics["mse"]:.2e}',
                'Cosine Similarity': f'{metrics["cosine_similarity"]:.4f}'
            })
        
        # Print results table
        print("\n📈 Model Performance Comparison")
        print(tabulate(results, headers='keys', tablefmt='grid'))
        
        # GPU vs CPU comparison if CUDA available
        if self.device == 'cuda' and torch.cuda.is_available():
            self._compare_gpu_cpu(onnx_path, input_array)
        
        return results
    
    def _compare_gpu_cpu(self, onnx_path: str, input_array: np.ndarray):
        """Compare GPU vs CPU inference performance."""
        print("\n🖥️  GPU vs CPU Comparison")
        
        # CPU inference
        cpu_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = cpu_session.get_inputs()[0].name
        
        start = time.time()
        for _ in range(100):
            _ = cpu_session.run(None, {input_name: input_array})
        cpu_time = (time.time() - start) / 100 * 1000
        
        # GPU inference
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            gpu_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
            
            start = time.time()
            for _ in range(100):
                _ = gpu_session.run(None, {input_name: input_array})
            gpu_time = (time.time() - start) / 100 * 1000
            
            speedup = cpu_time / gpu_time
            
            print(f"CPU Inference Time: {cpu_time:.1f} ms")
            print(f"GPU Inference Time: {gpu_time:.1f} ms")
            print(f"GPU Speedup: {speedup:.2f}x")
        else:
            print("GPU not available for comparison")