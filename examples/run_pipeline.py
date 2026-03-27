"""
Complete Pipeline Example
Run the full PyTorch → ONNX conversion, optimization, and benchmark pipeline.

Usage:
    python examples/run_pipeline.py --model resnet18
    python examples/run_pipeline.py --model mobilenet_v3_small --dynamo
    python examples/run_pipeline.py --model resnet50 --device cuda --num-runs 200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    Benchmark,
    ImageLoader,
    ModelConverter,
    ModelOptimizer,
    check_gpu_availability,
    create_output_directory,
)
from src.model_converter import ConversionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Run the complete conversion + optimization + benchmark pipeline."""
    logger.info("Starting PyTorch → ONNX pipeline")

    check_gpu_availability()
    output_dir = create_output_directory(args.output_dir)

    # ------------------------------------------------------------------
    # Step 1: Convert PyTorch → ONNX
    # ------------------------------------------------------------------
    logger.info("Step 1: Converting %s to ONNX (dynamo=%s)", args.model, args.dynamo)

    config = ConversionConfig(
        model_name=args.model,
        input_shape=(1, 3, args.input_size, args.input_size),
        pretrained=True,
        opset_version=args.opset,
        use_dynamo=args.dynamo,
    )
    converter = ModelConverter(config=config)
    onnx_path = converter.convert_to_onnx(
        output_path=os.path.join(output_dir, f"{args.model}.onnx")
    )

    # ------------------------------------------------------------------
    # Step 2: Optimize ONNX models
    # ------------------------------------------------------------------
    logger.info("Step 2: Optimizing ONNX models")

    optimizer = ModelOptimizer(onnx_path)
    simplified_path = optimizer.simplify()
    dynamic_quant_path = optimizer.dynamic_quantize()

    optimized_paths = [simplified_path, dynamic_quant_path]

    try:
        fp16_path = optimizer.fp16_quantize()
        optimized_paths.append(fp16_path)
    except ImportError as exc:
        logger.warning("%s  Skipping FP16 quantization.", exc)

    # ------------------------------------------------------------------
    # Step 3: Benchmark
    # ------------------------------------------------------------------
    logger.info("Step 3: Benchmarking all model variants (%d runs)", args.num_runs)

    benchmark = Benchmark(device=args.device, num_warmup=args.num_warmup)
    benchmark.compare_all_models(
        pytorch_model=converter.pytorch_model,
        onnx_path=onnx_path,
        optimized_paths=optimized_paths,
        input_shape=(1, 3, args.input_size, args.input_size),
        num_runs=args.num_runs,
    )

    # ------------------------------------------------------------------
    # Optional: real-image test
    # ------------------------------------------------------------------
    if args.test_real_images:
        logger.info("Loading CIFAR-10 samples for real-image inference check")
        loader = ImageLoader(input_size=(args.input_size, args.input_size))
        images, labels = loader.load_cifar10_sample(num_samples=args.batch_size)
        logger.info("Loaded %d CIFAR-10 images (shape %s)", images.shape[0], tuple(images.shape))

    # ------------------------------------------------------------------
    # Optional: dynamic-batch test
    # ------------------------------------------------------------------
    if args.test_dynamic_batch:
        logger.info("Testing dynamic batch sizes …")
        for batch_size in [1, 4, 8, 16]:
            dummy = converter.get_dummy_input(batch_size=batch_size)
            try:
                benchmark.measure_onnx_inference(onnx_path, dummy.numpy(), num_runs=5)
                logger.info("  batch_size=%d OK", batch_size)
            except Exception:
                logger.exception("  batch_size=%d FAILED", batch_size)

    logger.info("Pipeline finished.  Models saved to: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch → ONNX Conversion & Optimization Pipeline"
    )
    parser.add_argument("--model", default="resnet18",
                        help="torchvision model name (default: resnet18)")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Square input image size (default: 224)")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for saved models (default: models)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for benchmarking (default: cpu)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of timed inference iterations (default: 100)")
    parser.add_argument("--num-warmup", type=int, default=10,
                        help="Number of warmup iterations before timing (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for real-image test (default: 1)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--dynamo", action="store_true",
                        help="Use torch.onnx.export(dynamo=True) exporter (PyTorch 2.1+)")
    parser.add_argument("--test-real-images", action="store_true",
                        help="Test pipeline with real CIFAR-10 images")
    parser.add_argument("--test-dynamic-batch", action="store_true",
                        help="Verify ONNX model works with multiple batch sizes")

    main(parser.parse_args())
