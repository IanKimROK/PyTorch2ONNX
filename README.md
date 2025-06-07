# PyTorch to ONNX Model Conversion and Optimization Pipeline

A comprehensive pipeline for converting PyTorch models to ONNX format with various optimization techniques including quantization and performance benchmarking.

## Features

- ✅ PyTorch to ONNX conversion with dynamic batch size support
- ✅ Model optimization using `onnx-simplifier`
- ✅ Dynamic and Float16 quantization
- ✅ Performance benchmarking (inference time, accuracy, file size)
- ✅ GPU/CPU inference comparison
- ✅ Support for both dummy and real image inputs (CIFAR-10, ImageNet)
- ✅ Modular design for easy integration

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Additional Dependencies
```bash
# For FP16 conversion support
pip install onnxconverter-common

# For GPU support (choose based on your CUDA version)
pip install onnxruntime-gpu
```

### Install from Source
```bash
git clone https://github.com/yourusername/pytorch-onnx-pipeline.git
cd pytorch-onnx-pipeline
pip install -e .
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Run with default settings (ResNet18)
python examples/run_pipeline.py

# Run with custom model and options
python examples/run_pipeline.py --model resnet50 --device cuda --test-real-images --test-dynamic-batch

# Available options:
# --model: Model name (resnet18, resnet50, vgg16, mobilenet_v2, etc.)
# --input-size: Input image size (default: 224)
# --device: cpu or cuda
# --num-runs: Number of benchmark runs (default: 100)
# --test-real-images: Test with CIFAR-10 images
# --test-dynamic-batch: Test dynamic batch sizes
```

### Option 2: Use Standalone Script
```bash
# Install dependencies first
python run_complete_pipeline.py --install

# Run the pipeline
python run_complete_pipeline.py --model resnet18 --device cuda
```

### Option 3: Use as Python Module
```python
from src import ModelConverter, ModelOptimizer, Benchmark

# Step 1: Convert PyTorch to ONNX
converter = ModelConverter(model_name='resnet18')
onnx_path = converter.convert_to_onnx('models/resnet18.onnx')

# Step 2: Optimize the model
optimizer = ModelOptimizer(onnx_path)
simplified_path = optimizer.simplify()
dynamic_quant_path = optimizer.dynamic_quantize()
fp16_path = optimizer.fp16_quantize()

# Step 3: Benchmark all models
benchmark = Benchmark(device='cuda')  # or 'cpu'
results = benchmark.compare_all_models(
    pytorch_model=converter.pytorch_model,
    onnx_path=onnx_path,
    optimized_paths=[simplified_path, dynamic_quant_path, fp16_path]
)
```

## Project Structure

```
PyTorch2ONNX/
├── README.md
├── requirements.txt
├── run_complete_pipeline.py    # Standalone script with all features
├── src/
│   ├── __init__.py
│   ├── model_converter.py      # PyTorch to ONNX conversion
│   ├── model_optimizer.py      # ONNX optimization (simplify, quantize)
│   ├── benchmark.py            # Performance benchmarking
│   └── utils.py                # Helper utilities
├── examples/
│   └── run_pipeline.py         # Example usage script
├── models/                     # Output directory for converted models
└── data/                       # Dataset cache directory
```

## Supported Models

All models from `torchvision.models` are supported, including:
- ResNet family (resnet18, resnet34, resnet50, resnet101, resnet152)
- VGG family (vgg11, vgg13, vgg16, vgg19)
- MobileNet (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)
- EfficientNet (efficientnet_b0 to efficientnet_b7)
- DenseNet (densenet121, densenet169, densenet201)
- SqueezeNet (squeezenet1_0, squeezenet1_1)
- ShuffleNet (shufflenet_v2_x0_5, shufflenet_v2_x1_0)
- And more...

## Output Example

```
🚀 Starting PyTorch to ONNX Conversion Pipeline
============================================================

💻 System Information:
PyTorch version: 2.0.1
ONNX Runtime version: 1.16.0
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3090
Available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

📦 Step 1: Converting PyTorch model to ONNX
--------------------------------------------------
✅ Successfully converted resnet18 to ONNX: models/resnet18.onnx

🔧 Step 2: Optimizing ONNX models
--------------------------------------------------
✅ Model simplified and saved to: models/resnet18_simplified.onnx
✅ Dynamic quantization applied and saved to: models/resnet18_dynamic_quant.onnx
✅ FP16 conversion completed and saved to: models/resnet18_fp16.onnx

📊 Step 3: Benchmarking all models
--------------------------------------------------

📈 Model Performance Comparison
┌─────────────────────┬──────────────┬─────────────────┬──────────────┬─────────────┐
│ Model               │ File Size    │ Inference Time  │ MSE          │ Cosine Sim  │
├─────────────────────┼──────────────┼─────────────────┼──────────────┼─────────────┤
│ PyTorch (Original)  │ 44.7 MB      │ 12.5 ms        │ 0.0000       │ 1.0000      │
│ ONNX                │ 44.7 MB      │ 8.3 ms         │ 1.2e-07      │ 0.9999      │
│ ONNX Simplified     │ 44.6 MB      │ 8.1 ms         │ 1.2e-07      │ 0.9999      │
│ ONNX Dynamic Quant  │ 11.2 MB      │ 6.2 ms         │ 3.4e-05      │ 0.9998      │
│ ONNX FP16           │ 22.4 MB      │ 5.8 ms         │ 2.1e-06      │ 0.9999      │
└─────────────────────┴──────────────┴─────────────────┴──────────────┴─────────────┘

🖥️  GPU vs CPU Comparison
CPU Inference: 8.3 ms
GPU Inference: 2.1 ms
GPU Speedup: 4.0x

📦 Testing dynamic batch sizes
--------------------------------------------------
✅ Batch size 1: OK
✅ Batch size 4: OK
✅ Batch size 8: OK
✅ Batch size 16: OK

✨ Pipeline completed successfully!
📁 All models saved in: models/
```

## Advanced Usage

### Custom Model Integration

```python
import torch
import torch.nn as nn
from src import ModelConverter

# Define your custom model
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64 * 222 * 222, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create converter with custom model
model = CustomModel()
converter = ModelConverter.__new__(ModelConverter)
converter.pytorch_model = model
converter.input_shape = (1, 3, 224, 224)

# Convert to ONNX
onnx_path = converter.convert_to_onnx('models/custom_model.onnx')
```

### Batch Processing

```python
from src import ModelConverter, Benchmark

# Process multiple models
models_to_convert = ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']

for model_name in models_to_convert:
    print(f"\nProcessing {model_name}...")
    
    # Convert
    converter = ModelConverter(model_name=model_name)
    onnx_path = converter.convert_to_onnx(f'models/{model_name}.onnx')
    
    # Optimize
    optimizer = ModelOptimizer(onnx_path)
    optimizer.simplify()
    optimizer.dynamic_quantize()
    
    # Benchmark
    benchmark = Benchmark()
    results = benchmark.compare_all_models(
        pytorch_model=converter.pytorch_model,
        onnx_path=onnx_path,
        optimized_paths=[...]
    )
```

### Performance Tips

1. **GPU Acceleration**: Always use `--device cuda` when GPU is available for significant speedup
2. **Batch Size**: Test with realistic batch sizes for your use case
3. **Quantization**: Dynamic quantization provides the best size/accuracy tradeoff for most models
4. **Simplification**: Always simplify before quantization for better results

## Troubleshooting

### Common Issues

1. **CUDA/GPU not detected**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install appropriate ONNX Runtime GPU version
   pip install onnxruntime-gpu==1.16.0  # Match your CUDA version
   ```

2. **FP16 conversion fails**
   ```bash
   pip install onnxconverter-common
   ```

3. **Out of memory errors**
   - Reduce batch size
   - Use CPU for conversion, GPU for inference
   - Clear cache between runs

4. **Dynamic axes errors**
   - Ensure your model supports variable batch sizes
   - Some models may need fixed input sizes

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
converter = ModelConverter(model_name='resnet18')
torch.onnx.export(..., verbose=True)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.