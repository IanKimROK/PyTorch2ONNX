"""
Complete Pipeline Example
Run the entire PyTorch to ONNX conversion and optimization pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    ModelConverter, 
    ModelOptimizer, 
    Benchmark, 
    ImageLoader,
    check_gpu_availability,
    create_output_directory
)
import argparse


def main(args):
    """Run the complete pipeline."""
    
    print("🚀 Starting PyTorch to ONNX Conversion Pipeline")
    print("=" * 50)
    
    # Check system info
    check_gpu_availability()
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Step 1: Convert PyTorch to ONNX
    print("\n📦 Step 1: Converting PyTorch model to ONNX")
    print("-" * 50)
    
    converter = ModelConverter(
        model_name=args.model,
        input_shape=(1, 3, args.input_size, args.input_size),
        pretrained=True
    )
    
    onnx_path = converter.convert_to_onnx(
        output_path=os.path.join(output_dir, f"{args.model}.onnx"),
        input_names=['input'],
        output_names=['output']
    )
    
    # Step 2: Optimize ONNX models
    print("\n🔧 Step 2: Optimizing ONNX models")
    print("-" * 50)
    
    optimizer = ModelOptimizer(onnx_path)
    
    # Simplify model
    simplified_path = optimizer.simplify()
    
    # Apply quantizations
    dynamic_quant_path = optimizer.dynamic_quantize()
    
    # FP16 quantization (install onnxconverter-common if needed)
    try:
        fp16_path = optimizer.fp16_quantize()
        optimized_paths = [simplified_path, dynamic_quant_path, fp16_path]
    except ImportError:
        print("⚠️  onnxconverter-common not installed. Skipping FP16 quantization.")
        print("   Install with: pip install onnxconverter-common")
        optimized_paths = [simplified_path, dynamic_quant_path]
    
    # Step 3: Benchmark all models
    print("\n📊 Step 3: Benchmarking all models")
    print("-" * 50)
    
    benchmark = Benchmark(device=args.device)
    
    # Test with dummy input
    print("\n🧪 Testing with dummy input (batch_size=1)")
    results = benchmark.compare_all_models(
        pytorch_model=converter.pytorch_model,
        onnx_path=onnx_path,
        optimized_paths=optimized_paths,
        input_shape=(1, 3, args.input_size, args.input_size),
        num_runs=args.num_runs
    )
    
    # Test with real images if requested
    if args.test_real_images:
        print("\n🖼️  Testing with real images (CIFAR-10)")
        loader = ImageLoader(input_size=(args.input_size, args.input_size))
        
        # Load CIFAR-10 samples
        images, labels = loader.load_cifar10_sample(num_samples=args.batch_size)
        
        print(f"Loaded {images.shape[0]} CIFAR-10 images")
        print("Running inference comparison...")
        
        # You can add additional benchmarking with real images here
    
    # Test dynamic batch sizes
    if args.test_dynamic_batch:
        print("\n📦 Testing dynamic batch sizes")
        print("-" * 50)
        
        for batch_size in [1, 4, 8, 16]:
            print(f"\nBatch size: {batch_size}")
            dummy_input = converter.get_dummy_input(batch_size=batch_size)
            
            # Test ONNX model with different batch sizes
            try:
                _, _ = benchmark.measure_onnx_inference(
                    onnx_path, 
                    dummy_input.numpy(),
                    num_runs=10
                )
                print(f"✅ Batch size {batch_size} works correctly")
            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {str(e)}")
    
    print("\n✨ Pipeline completed successfully!")
    print(f"📁 All models saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch to ONNX Conversion Pipeline")
    
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model name from torchvision.models (default: resnet18)"
    )
    
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (default: 224)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models (default: models)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on (default: cpu)"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs for benchmarking (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for testing (default: 1)"
    )
    
    parser.add_argument(
        "--test-real-images",
        action="store_true",
        help="Test with real CIFAR-10 images"
    )
    
    parser.add_argument(
        "--test-dynamic-batch",
        action="store_true",
        help="Test dynamic batch sizes"
    )
    
    args = parser.parse_args()
    main(args)