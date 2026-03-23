# PyTorch to ONNX Model Conversion and Optimization Pipeline

A comprehensive pipeline for converting PyTorch models to ONNX format with various optimization and compression techniques.

## Features

- PyTorch → ONNX conversion with dynamic batch size support
- **Dynamo export** – `torch.onnx.export(dynamo=True)` for full graph capture (PyTorch 2.1+)
- Model graph simplification via `onnx-simplifier`
- Dynamic INT8 quantization and FP16 conversion
- **ORT graph optimization** – `ORT_ENABLE_ALL` applied automatically on every session
- Performance benchmarking (inference time, file size, MSE, cosine similarity)
- GPU / CPU inference comparison
- `ConversionConfig` dataclass for type-safe, IDE-friendly configuration
- Modular design (`src/`) with a thin standalone entry-point (`run_complete_pipeline.py`)
- Pytest test suite (offline – no weight downloads required)

## Installation

```bash
pip install -r requirements.txt
```

> **GPU support**: `onnxruntime` and `onnxruntime-gpu` cannot be installed simultaneously.
> For CUDA inference, uninstall `onnxruntime` first, then:
> ```bash
> pip uninstall onnxruntime
> pip install onnxruntime-gpu          # picks the right version for your CUDA
> ```

## Quick Start

### Option 1: Standalone script

```bash
# Default – ResNet18, CPU, opset 17
python run_complete_pipeline.py

# ResNet50 on GPU with Dynamo exporter
python run_complete_pipeline.py --model resnet50 --device cuda --dynamo

# Install / verify all dependencies
python run_complete_pipeline.py --install
```

### Option 2: Modular example

```bash
python examples/run_pipeline.py --model mobilenet_v3_small --device cuda \
    --dynamo --opset 17 --num-runs 200 --test-dynamic-batch
```

**All CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `resnet18` | Any `torchvision.models` name |
| `--input-size` | `224` | Square input image side length |
| `--output-dir` | `models` | Directory for saved ONNX files |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--num-runs` | `100` | Timed inference iterations |
| `--num-warmup` | `10` | Warmup iterations before timing |
| `--opset` | `17` | ONNX opset version |
| `--dynamo` | off | Use `torch.onnx.export(dynamo=True)` |
| `--test-real-images` | off | Load 5 CIFAR-10 samples |
| `--test-dynamic-batch` | off | Verify batch sizes 1 / 4 / 8 / 16 |

### Option 3: Python API

```python
from src import ConversionConfig, ModelConverter, ModelOptimizer, Benchmark

# --- Step 1: Configure and convert ---
config = ConversionConfig(
    model_name="resnet50",
    input_shape=(1, 3, 224, 224),
    pretrained=True,
    opset_version=17,
    use_dynamo=False,         # set True for PyTorch 2.1+ dynamo path
)
converter = ModelConverter(config=config)
onnx_path = converter.convert_to_onnx("models/resnet50.onnx")

# --- Step 2: Optimise ---
optimizer = ModelOptimizer(onnx_path)
simplified_path   = optimizer.simplify()
int8_path         = optimizer.dynamic_quantize()
fp16_path         = optimizer.fp16_quantize()   # requires onnxconverter-common

# Optionally get an ORT session with ORT_ENABLE_ALL graph optimization
session = optimizer.create_optimized_session()

# --- Step 3: Benchmark ---
benchmark = Benchmark(device="cpu", num_warmup=10)
results = benchmark.compare_all_models(
    pytorch_model=converter.pytorch_model,
    onnx_path=onnx_path,
    optimized_paths=[simplified_path, int8_path, fp16_path],
)
```

### Custom model

Pass your own `torch.nn.Module` directly – no torchvision dependency required:

```python
import torch, torch.nn as nn
from src.model_converter import ConversionConfig, ModelConverter

class MyModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=[2, 3])

# Bypass the torchvision loader by setting pytorch_model manually
config = ConversionConfig(input_shape=(1, 3, 224, 224), pretrained=False)
converter = ModelConverter.__new__(ModelConverter)
converter.config = config
converter.pytorch_model = MyModel().eval()

converter.convert_to_onnx("models/my_model.onnx")
```

## Project Structure

```
PyTorch2ONNX/
├── README.md
├── requirements.txt
├── pyproject.toml                  # pytest configuration
├── run_complete_pipeline.py        # standalone entry-point
├── src/
│   ├── __init__.py                 # exports ConversionConfig + all public classes
│   ├── model_converter.py          # ConversionConfig dataclass + ModelConverter
│   ├── model_optimizer.py          # simplify / INT8 / FP16 / ORT session
│   ├── benchmark.py                # latency, file size, MSE, cosine similarity
│   └── utils.py                    # ImageLoader, check_gpu_availability
├── examples/
│   └── run_pipeline.py             # full pipeline with all CLI flags
├── tests/
│   ├── conftest.py                 # shared fixtures (TinyModel, no downloads)
│   ├── test_converter.py
│   ├── test_optimizer.py
│   ├── test_benchmark.py
│   └── test_utils.py
├── models/                         # ONNX output files (git-ignored)
└── data/                           # dataset cache (git-ignored)
```

## Supported Models

Any model in `torchvision.models` works out of the box (loaded via `get_model` /
`get_model_weights` – the modern weights API introduced in torchvision 0.13):

```
resnet18/34/50/101/152 · vgg11/13/16/19 · mobilenet_v2/v3_small/v3_large
efficientnet_b0…b7 · densenet121/169/201 · squeezenet1_0/1_1
shufflenet_v2_x0_5/x1_0 · convnext_tiny/small/base/large · swin_t/s/b …
```

## Running Tests

```bash
# All tests (no network required)
pytest

# Specific module
pytest tests/test_benchmark.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

## Performance Tips

1. **Simplify before quantizing** – cleaner graphs yield better INT8 calibration
2. **`ORT_ENABLE_ALL`** is applied automatically; no extra step needed
3. **Dynamo export** (`--dynamo`) is recommended for models with complex control flow
4. **FP16** gives the best latency/size tradeoff on CUDA devices with tensor cores
5. **Increase `--num-warmup`** (e.g. `--num-warmup 20`) for more stable GPU timings

## Troubleshooting

**CUDA not detected**
```bash
python -c "import torch; print(torch.cuda.is_available())"
pip install onnxruntime-gpu   # after uninstalling onnxruntime
```

**FP16 conversion fails**
```bash
pip install onnxconverter-common   # already in requirements.txt
```

**Dynamo export errors**
- Requires PyTorch ≥ 2.1; fall back to the default legacy path by omitting `--dynamo`

**Out of memory**
- Reduce batch size or run conversion on CPU, inference on GPU

**Enable DEBUG logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT License – see [LICENSE](LICENSE) for details.
