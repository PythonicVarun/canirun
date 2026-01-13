# canirun

A lightweight CLI to estimate hardware requirements and quantization compatibility for Hugging Face models.

[![PyPI version](https://badge.fury.io/py/canirun.svg)](https://badge.fury.io/py/canirun)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
