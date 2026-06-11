# Isildur

![CI](https://github.com/Enotrium/Isildur/actions/workflows/ci.yml/badge.svg)

**Turn Any Neural Network into a Vector Symbolic Architecture (VSA/HDC) — Run AI Inference in Seconds Without Backpropagation**

Isildur is the bridge between neural networks and Hyperdimensional Computing. It converts any trained neural network into balanced binary hypervectors (~10,000-bit), enables ultra-low-power inference via in-memory Hamming distance computation on neuromorphic FPGAs, and fuses multiple models via HDC consensus — all without retraining or backpropagation.

## The Vision

VSA/HDC lacks the dedicated, reconfigurable circuitry — the "transformer equivalent for HDC" — that would enable simple, ultra-low-power hypervector operations (XOR, bundling, permutation on ~10k-bit random binary vectors) at near-zero marginal cost.

Without this in-memory Hamming distance, HDC's proven advantages in efficiency, noise robustness, one-shot learning, native multimodality, and seamless coexistence with existing NNs remain underutilized. Building it is the key unlock: it lets us zoom out to broad patterns across high-dimensional spaces, map old models into HDC without replacement, and drive inference costs toward zero while unlocking the next era of efficient, general intelligence.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ISILDUR PIPELINE                              │
│                                                                         │
│  Any Trained NN ──────► HDC Converter ──────► Binary HV (~10k bits)     │
│  (ResNet, ViT,        (LayerBinarizer +    (Balanced ±1 bipolar)        │
│   GPT, CNN, MLP)       HVComposer)                                      │
│       │                     │                     │                     │
│       │              ┌──────┴──────┐       ┌──────┴──────────────┐      │
│       │              │  Binarize   │       │  HDC Operations     │      │
│       │              │  activations│       │  • XOR (Binding)    │      │
│       │              │  + weights  │       │  • Bundling         │      │
│       │              └─────────────┘       │  • Permutation      │      │
│       │                                    │  • Similarity       │      │
│       │                                    └──────┬──────────────┘      │
│       │                                           │                     │
│       │                    ┌──────────────────────┘                     │
│       │                    ▼                                            │
│       │           ┌───────────────────┐                                 │
│       │           │  CIM Associative   │  ← Computing-in-Memory         │
│       │           │  Memory (TCAM)     │     Hamming distance           │
│       │           │  ~0.5pJ per search │     on neuromorphic FPGA       │
│       │           └────────┬──────────┘                                 │
│       │                    │                                            │
│       │                    ▼                                            │
│       │           ┌───────────────────┐                                 │
│       │           │  HDC Consensus     │  ← Model Fusion via            │
│       │           │  (HD-Glue)         │   Hyperdimensional Binding     │
│       │           └────────┬──────────┘                                 │
│       │                    │                                            │
│       │                    ▼                                            │
│       │            ╔═══════════════════╗                                │
│       │            ║  FPGA / ASIC      ║  ← Neuromorphic Inference      │
│       │            ║  XCZU9EG / TSMC   ║     100MHz, ~15mW, ~100ns/tick │
│       │            ╚═══════════════════╝                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Capabilities

| Capability | Description |
|-----------|-------------|
| **NN → HV Conversion** | Any PyTorch model → balanced binary hypervector in one forward pass |
| **In-Memory Inference** | TCAM-based CIM Hamming distance at ~0.5pJ per search |
| **One-Shot Learning** | New classes added with single-sample bundling |
| **Model Fusion** | HD-Glue consensus across multiple models without retraining |
| **Noise Robustness** | Random bit-flip tolerance up to ~30% corruption |
| **Multimodality** | Native binding of modalities via hypervector operations |
| **FPGA Synthesis** | HLS C++ and Verilog templates for neuromorphic FPGA deployment |
| **12hr→Seconds** | Convert trained models and run inference in seconds |

## Quick Start

```bash
# Install
pip install -e .

# Convert a ResNet18 to hypervector
isildur convert --model resnet18 --hv-dim 10000 --save model.hv

# Compare two saved hypervectors
isildur compare model_a.hv model_b.hv

# Run HDC inference with CIM
isildur infer --hv model.hv --input sample.pt --hardware cim

# Export FPGA HLS for synthesis
isildur export-fpga --hv-dim 10000 --target xczu9eg

# Fuse multiple models via HDC consensus
isildur fuse --models model_a.hv model_b.hv model_c.hv --output fused.hv
```

## Python API

```python
import torch
from isildur import Hypervector, CIMAssociativeMemory, HDGlue
from isildur.nn_to_hdc import model_to_hv

# Convert any NN to hypervector
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
hv = model_to_hv(model, hv_dim=10000)

# In-memory inference
cim = CIMAssociativeMemory(n_classes=10, hv_dim=10000)
cim.encode(training_data, labels)
prediction, distance = cim.infer(hv)

# Fuse multiple models
fusion = HDGlue(n_models=3, n_classes=10, dim=10000)
fusion.train_consensus(model_0, class_0)
result = fusion.predict(all_model_outputs)
```

## From Paper to Silicon

Isildur provides complete tooling from research to tapeout:

1. **Software Simulation** — Pure PyTorch HDC operations with CIM simulation
2. **Fixed-Point Quantization** — INT8/INT16 quantized HDC for hardware
3. **HLS C++ Templates** — Systolic bind arrays, CIM Hamming, HV ops for Vitis HLS
4. **Verilog Templates** — XOR, bundle, and associative memory RTL modules
5. **FPGA Build Scripts** — Vitis HLS synthesis configurations for XCZU9EG
6. **ASIC Path** — TSMC 28nm estimates (~1mm² for complete HDC accelerator)

## Resource Estimates (XCZU9EG)

| Module | LUTs | DSPs | BRAM | Power |
|--------|------|------|------|-------|
| HV XOR/Bind Array | ~2K | 0 | 2 | ~3mW |
| CIM Hamming (TCAM) | ~3K | 0 | 4 | ~5mW |
| Associative Memory | ~1K | 0 | 2 | ~2mW |
| Systolic Bundle | ~2K | 64 | 4 | ~5mW |
| **Total** | **~8K** | **64** | **12** | **~15mW** |

At 100MHz with 5% activity: **~100ns per inference** on 10,000-bit hypervectors.


## License

Confidential — Enotrium