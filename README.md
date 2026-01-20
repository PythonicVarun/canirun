# canirun

A lightweight CLI to estimate hardware requirements and quantization compatibility for Hugging Face models.

[![codecov](https://codecov.io/github/PythonicVarun/canirun/branch/master/graph/badge.svg?token=FHPEETQPA7)](https://codecov.io/github/PythonicVarun/canirun)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/canirun)](https://pypi.org/project/canirun)
[![PyPI - Version](https://img.shields.io/pypi/v/canirun)
](https://pypi.org/project/canirun)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> [!NOTE]
> Currently optimized for standard Transformer architectures (Llama, Mistral, Gemma, BERT). MoE and custom architectures may have experimental support.

## Key Features

- **Hardware Detection**: Automatically detects your CPU/GPU and available VRAM/RAM.
- **Memory Estimation**: Estimates the memory required to run a given Hugging Face model.
- **Quantization Analysis**: Checks compatibility for different quantization levels (e.g., 4-bit, 8-bit, 16-bit).
- **Simple CLI & API**: Easy to use from the command line or integrate into your Python projects.

## Installation

You can install `canirun` using pip:

```bash
pip install canirun
```

## CLI Usage

The `canirun` command allows you to quickly check a model from your terminal.

```bash
canirun <model_id> [OPTIONS]
```

### Example

Let's check if `meta-llama/Meta-Llama-3-8B` can run on the local hardware:

```bash
canirun meta-llama/Meta-Llama-3-8B --ctx 4096
```

This will produce a report like this:

```
 🔍 ANALYSIS REPORT: meta-llama/Meta-Llama-3-8B
 Context Length  : 4096
 Device          : NVIDIA GeForce RTX 3090
 VRAM / RAM      : 24.0 GB / 64.0 GB

╒════════════════╤══════════════╤════════════╤════════════════════════╕
│ Quantization   │   Total Est. │   KV Cache │ Compatibility          │
╞════════════════╪══════════════╪════════════╪════════════════════════╡
│ FP16           │     16.96 GB │  512.00 MB │ ✅ GPU                 │
├────────────────┼──────────────┼────────────┼────────────────────────┤
│ INT8           │      9.48 GB │  512.00 MB │ ✅ GPU                 │
├────────────────┼──────────────┼────────────┼────────────────────────┤
│ 4-bit          │      6.30 GB │  512.00 MB │ ✅ GPU                 │
├────────────────┼──────────────┼────────────┼────────────────────────┤
│ 2-bit          │      4.34 GB │  512.00 MB │ ✅ GPU                 │
╘════════════════╧══════════════╧════════════╧════════════════════════╛
```

## API Usage

You can also use `canirun` programmatically in your Python code.

```python
from canirun import canirun

model_id = "mistralai/Mistral-7B-v0.1"

# Analyze the model
result = canirun(model_id, context_length=2048)

if result and result.is_supported:
    print(f"'{model_id}' is supported on your hardware!")

    # Get the detailed report
    report = result.report()
    for quant_result in report:
        print(f"- {quant_result['quant']}: {quant_result['status']}")
else:
    print(f"'{model_id}' is not supported on your hardware.")

```

## How It Works

`canirun` works by:
1. Fetching the model's configuration from the Hugging Face Hub.
2. Calculating the memory required for the model's parameters.
3. Estimating the size of the KV cache based on the context length and model architecture.
4. Comparing the estimated memory requirements with your system's available VRAM (if a GPU is detected) or RAM.

The tool checks for different levels of quantization to see if a smaller, quantized version of the model could fit.

## Development

This project maintains strict code quality standards:

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

- **Formatter**: Black
- **Linter**: Ruff
- **Type Checking**: MyPy (Strict)
- **Docstrings**: Google Style

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
