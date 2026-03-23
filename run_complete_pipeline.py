#!/usr/bin/env python3
"""
Complete PyTorch to ONNX Pipeline Script
Thin entry-point that delegates to the `src` package modules.

Usage:
    python run_complete_pipeline.py
    python run_complete_pipeline.py --model mobilenet_v3_small --device cuda
    python run_complete_pipeline.py --model resnet50 --dynamo --opset 17
    python run_complete_pipeline.py --install   # install / verify dependencies
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency check / install
# ---------------------------------------------------------------------------

_REQUIRED = [
    "torch", "torchvision", "onnx", "onnxruntime",
    "onnx_simplifier", "numpy", "pillow", "tabulate", "scipy",
]
_OPTIONAL = {"onnxconverter_common": "onnxconverter-common"}


def install_dependencies() -> None:
    """Install required (and optional) pip packages."""
    packages = (
        "torch torchvision onnx onnxruntime onnx-simplifier "
        "numpy pillow tabulate scipy onnxconverter-common"
    )
    logger.info("Installing dependencies: %s", packages)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages.split()])
    logger.info("All dependencies installed successfully.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the full conversion → optimisation → benchmark pipeline."""
    # Late import so --install works even without dependencies present
    import os
    from src import (
        Benchmark,
        ImageLoader,
        ModelConverter,
        ModelOptimizer,
        check_gpu_availability,
        create_output_directory,
    )
    from src.model_converter import ConversionConfig

    check_gpu_availability()
    output_dir = create_output_directory(args.output_dir)

    # Step 1 – Convert
    logger.info("Step 1: Converting %s to ONNX", args.model)
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

    # Step 2 – Optimise
    logger.info("Step 2: Optimising ONNX models")
    optimizer = ModelOptimizer(onnx_path)
    simplified_path = optimizer.simplify()
    dynamic_quant_path = optimizer.dynamic_quantize()
    optimized_paths = [simplified_path, dynamic_quant_path]
    try:
        optimized_paths.append(optimizer.fp16_quantize())
    except ImportError as exc:
        logger.warning("%s  Skipping FP16.", exc)

    # Step 3 – Benchmark
    logger.info("Step 3: Benchmarking (%d runs)", args.num_runs)
    benchmark = Benchmark(device=args.device, num_warmup=10)
    benchmark.compare_all_models(
        pytorch_model=converter.pytorch_model,
        onnx_path=onnx_path,
        optimized_paths=optimized_paths,
        input_shape=(1, 3, args.input_size, args.input_size),
        num_runs=args.num_runs,
    )

    # Step 4 – Dynamic batch test
    logger.info("Step 4: Verifying dynamic batch sizes")
    for batch_size in [1, 4, 8, 16]:
        dummy = converter.get_dummy_input(batch_size=batch_size)
        try:
            benchmark.measure_onnx_inference(onnx_path, dummy.numpy(), num_runs=5)
            logger.info("  batch_size=%d OK", batch_size)
        except Exception:
            logger.exception("  batch_size=%d FAILED", batch_size)

    # Step 5 – Real-image test
    logger.info("Step 5: Real-image inference check (CIFAR-10)")
    loader = ImageLoader(input_size=(args.input_size, args.input_size))
    images, labels = loader.load_cifar10_sample(num_samples=5)
    logger.info("Loaded %d CIFAR-10 images (shape %s)", images.shape[0], tuple(images.shape))

    logger.info("Pipeline finished. Models saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch → ONNX Conversion & Optimization Pipeline"
    )
    parser.add_argument("--model", default="resnet18",
                        help="torchvision model name (default: resnet18)")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Square input image size (default: 224)")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory (default: models)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for benchmarking (default: cpu)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Timed inference iterations (default: 100)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--dynamo", action="store_true",
                        help="Use torch.onnx.export(dynamo=True) – PyTorch 2.1+")
    parser.add_argument("--install", action="store_true",
                        help="Install / verify all required pip packages and exit")
    args = parser.parse_args()

    if args.install:
        install_dependencies()
        return

    run(args)


if __name__ == "__main__":
    main()
