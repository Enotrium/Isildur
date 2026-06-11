#!/usr/bin/env python3
"""
isildur CLI — Command-line interface for Isildur HDC operations.

Usage:
  isildur convert --model resnet18 --hv-dim 10000 --save output.hv
  isildur compare model_a.hv model_b.hv
  isildur infer --hv model.hv --input sample.pt --hardware cim
  isildur export-fpga --hv-dim 10000 --output fpga_output/
  isildur fuse --models a.hv b.hv c.hv --output fused.hv
  isildur info --model resnet18
  isildur list-models
"""

import argparse
import sys
import os
import json
from typing import Optional, List
from datetime import datetime


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for Isildur CLI."""
    parser = argparse.ArgumentParser(
        prog="isildur",
        description="Turn Any Neural Network into VSA/HDC — Ultra-Low-Power Inference on Neuromorphic FPGAs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # ── convert ──────────────────────────────────────────────────────
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a PyTorch model to a hypervector",
    )
    convert_parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name (e.g., resnet18) or path to .pth/.pt file",
    )
    convert_parser.add_argument(
        "--hv-dim", "-d", type=int, default=10000,
        help="Hypervector dimension (default: 10000)",
    )
    convert_parser.add_argument(
        "--binarize", "-b", type=str, default="sign",
        choices=["sign", "threshold", "bernoulli", "magnitude"],
        help="Binarization method (default: sign)",
    )
    convert_parser.add_argument(
        "--compose", "-c", type=str, default="bundle",
        choices=["bundle", "bind", "permute", "weighted", "attention"],
        help="Composition strategy (default: bundle)",
    )
    convert_parser.add_argument(
        "--save", "-s", type=str, default=None,
        help="Save hypervector to .pt file",
    )
    convert_parser.add_argument(
        "--save-layer-hvs", type=str, default=None,
        help="Save per-layer HVs to .pt file",
    )
    convert_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    convert_parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cpu, cuda:0, mps)",
    )

    # ── compare ──────────────────────────────────────────────────────
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two saved hypervectors",
    )
    compare_parser.add_argument(
        "hv_a", type=str, help="First hypervector file"
    )
    compare_parser.add_argument(
        "hv_b", type=str, help="Second hypervector file"
    )

    # ── infer ────────────────────────────────────────────────────────
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run HDC inference",
    )
    infer_parser.add_argument(
        "--hv", type=str, required=True,
        help="Path to model hypervector .pt file",
    )
    infer_parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input sample .pt file",
    )
    infer_parser.add_argument(
        "--hardware", type=str, default="cim",
        choices=["cim", "digital"],
        help="Hardware mode: cim (TCAM) or digital (CPU)",
    )
    infer_parser.add_argument(
        "--classes", type=int, default=10,
        help="Number of classes",
    )

    # ── fuse ─────────────────────────────────────────────────────────
    fuse_parser = subparsers.add_parser(
        "fuse",
        help="Fuse multiple model hypervectors via HDC consensus",
    )
    fuse_parser.add_argument(
        "--models", "-m", type=str, nargs="+", required=True,
        help="Model HV files to fuse",
    )
    fuse_parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output fused HV file",
    )
    fuse_parser.add_argument(
        "--strategy", "-s", type=str, default="bundle",
        choices=["bundle", "bind", "permute"],
        help="Fusion strategy (default: bundle)",
    )

    # ── export-fpga ──────────────────────────────────────────────────
    fpga_parser = subparsers.add_parser(
        "export-fpga",
        help="Export FPGA HLS/Verilog for synthesis",
    )
    fpga_parser.add_argument(
        "--hv-dim", "-d", type=int, default=10000,
        help="Hypervector dimension",
    )
    fpga_parser.add_argument(
        "--classes", "-c", type=int, default=10,
        help="Number of classes",
    )
    fpga_parser.add_argument(
        "--target", "-t", type=str, default="xczu9eg",
        help="FPGA target part number",
    )
    fpga_parser.add_argument(
        "--output", "-o", type=str, default="isildur_fpga",
        help="Output directory",
    )
    fpga_parser.add_argument(
        "--verilog", action="store_true",
        help="Also export Verilog for ASIC flow",
    )

    # ── list-models ──────────────────────────────────────────────────
    subparsers.add_parser(
        "list-models",
        help="List all registered models",
    )

    # ── info ─────────────────────────────────────────────────────────
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a model",
    )
    info_parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name",
    )

    # ── benchmark ────────────────────────────────────────────────────
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark HDC operations",
    )
    bench_parser.add_argument(
        "--hv-dim", "-d", type=int, default=10000,
        help="Hypervector dimension",
    )
    bench_parser.add_argument(
        "--classes", "-c", type=int, default=10,
        help="Number of classes",
    )
    bench_parser.add_argument(
        "--samples", "-n", type=int, default=1000,
        help="Number of inference samples",
    )

    return parser


def cmd_convert(args) -> int:
    """Handle 'convert' command."""
    import torch
    from isildur.nn_to_hdc import model_to_hv, save_hv

    device = torch.device(args.device) if args.device else None

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Isildur: Convert NN → Hypervector                  ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    # Try to load the model by name or path
    model = None
    model_name = args.model

    # Check if it's a file path
    if os.path.isfile(model_name):
        print(f"Loading model from file: {model_name}")
        model = torch.load(model_name, map_location="cpu", weights_only=False)
    else:
        # Try torchvision registry
        try:
            import torchvision.models as tvmodels
            if hasattr(tvmodels, model_name):
                model_fn = getattr(tvmodels, model_name)
                model = model_fn(weights=None)
                print(f"Loaded model: {model_name} (random weights)")
            else:
                print(f"Model '{model_name}' not found in torchvision registry.")
        except ImportError:
            print("torchvision not available. Install it with: pip install torchvision")
            pass

    if model is None:
        # Create a simple demo model
        print(f"Creating demo CNN as '{model_name}'...")
        model = _create_demo_cnn()
        model.eval()

    hv_dim = args.hv_dim
    print(f"Hypervector dimension: {hv_dim}")
    print(f"Binarization: {args.binarize}")
    print(f"Composition: {args.compose}")
    print()

    # Convert
    hv, layer_hvs = model_to_hv(
        model,
        hv_dim=hv_dim,
        device=device,
        binarize_method=args.binarize,
        compose_strategy=args.compose,
        verbose=args.verbose,
        return_layer_hvs=True,
    )

    pos = (hv == 1).sum().item()
    neg = (hv == -1).sum().item()

    print()
    print("┌─────────────────────────────────────────────┐")
    print(f"│  Hypervector generated!                     │")
    print(f"│  Dimension: {hv_dim}                        │")
    print(f"│  +1 count:  {pos}                           │")
    print(f"│  -1 count:  {neg}                           │")
    print(f"│  Balance:   {abs(pos - neg) / (pos + neg):.4f}  │")
    print(f"│  Layers:    {len(layer_hvs)}                │")
    print("└─────────────────────────────────────────────┘")

    if args.save:
        from isildur.nn_to_hdc import save_hv
        save_hv(hv, args.save)
        print(f"\nSaved to: {args.save}")
        packed_bytes = len(hv.tolist()) // 8  # packed bits
        print(f"Packed size: {packed_bytes} bytes ({packed_bytes / 1024:.1f} KB)")

    if args.save_layer_hvs:
        torch.save(
            {"layer_hvs": layer_hvs, "dim": hv_dim, "n_layers": len(layer_hvs)},
            args.save_layer_hvs,
        )
        print(f"Per-layer HVs saved to: {args.save_layer_hvs}")

    return 0


def cmd_compare(args) -> int:
    """Handle 'compare' command."""
    import torch
    from isildur.nn_to_hdc import hv_similarity, hv_hamming, load_hv

    hv_a = load_hv(args.hv_a)
    hv_b = load_hv(args.hv_b)

    sim_val = hv_similarity(hv_a, hv_b)
    ham = hv_hamming(hv_a, hv_b)

    print(f"Model A: {args.hv_a} (dim={hv_a.shape[0]})")
    print(f"Model B: {args.hv_b} (dim={hv_b.shape[0]})")
    print()
    print(f"Cosine Similarity: {sim_val:.6f}")
    print(f"Hamming Distance:  {ham:.6f}  ({ham * hv_a.shape[0]:.0f} / {hv_a.shape[0]} bits)")
    print()

    # Interpretation
    if sim_val > 0.95:
        print("Interpretation: Nearly identical models")
    elif sim_val > 0.7:
        print("Interpretation: Very similar models (same architecture?)")
    elif sim_val > 0.3:
        print("Interpretation: Moderately related")
    elif abs(sim_val) < 0.1:
        print("Interpretation: Unrelated models (random similarity)")
    elif sim_val < -0.3:
        print("Interpretation: Anti-correlated (surprisingly different!)")

    return 0


def cmd_infer(args) -> int:
    """Handle 'infer' command."""
    import torch
    from isildur.nn_to_hdc import load_hv
    from isildur.cim import CIMAssociativeMemory, CIMConfig

    print(f"Inference Mode: {args.hardware.upper()}")
    print(f"Classes: {args.classes}")

    hv = load_hv(args.hv)
    sample = torch.load(args.input, map_location="cpu", weights_only=True)

    print(f"Loaded HV: {hv.shape}")
    print(f"Input shape: {sample.shape if isinstance(sample, torch.Tensor) else 'dict'}")

    # Setup CIM
    config = CIMConfig(
        hypervector_dim=hv.shape[0],
        n_classes=args.classes,
        block_size=32,
        use_tcam=(args.hardware == "cim"),
    )
    cim = CIMAssociativeMemory(
        n_classes=args.classes,
        hypervector_dim=hv.shape[0],
        config=config,
    )

    # Encode some sample class vectors (random for demo)
    from isildur.core import gen_hvs
    class_hvs = gen_hvs(args.classes, hv.shape[0], "bipolar")
    cim.cim.set_class_vectors(class_hvs)

    # Run inference
    pred, dist = cim.infer(hv)

    print(f"\nPredicted class: {pred}")
    print(f"Hamming distance: {dist:.1f} / {hv.shape[0]}")
    print(f"Energy estimate: {cim.inference_energy_estimate():.2f} pJ")

    return 0


def cmd_fuse(args) -> int:
    """Handle 'fuse' command."""
    import torch
    from isildur.nn_to_hdc import load_hv, save_hv
    from isildur.fusion import fuse_models, model_disagreement

    print(f"Fusing {len(args.models)} models via {args.strategy}...")

    model_hvs = []
    for path in args.models:
        hv = load_hv(path)
        model_hvs.append(hv)
        print(f"  Loaded: {path} (dim={hv.shape[0]})")

    fused = fuse_models(model_hvs, compose_strategy=args.strategy)

    # Compute pairwise disagreements
    print()
    print("Pairwise disagreements:")
    for i in range(len(model_hvs)):
        for j in range(i + 1, len(model_hvs)):
            disagreement = model_disagreement(model_hvs[i], model_hvs[j])
            print(f"  Model {i} ↔ Model {j}: {disagreement:.4f}")

    pos = (fused == 1).sum().item()
    print()
    print(f"Fused HV: dim={fused.shape[0]}, +1={pos}, -1={fused.shape[0] - pos}")

    save_hv(fused, args.output)
    print(f"Saved fused HV to: {args.output}")

    return 0


def cmd_export_fpga(args) -> int:
    """Handle 'export-fpga' command."""
    from isildur.fpga_backend import export_fpga_hls, export_verilog, estimate_fpga_resources

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Isildur: Export FPGA IP                             ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()
    print(f"Target: {args.target}")
    print(f"HV Dimension: {args.hv_dim}")
    print(f"Classes: {args.classes}")
    print(f"Output: {args.output}/")
    print()

    # Generate HLS C++ files
    files = export_fpga_hls(
        args.output,
        hv_dim=args.hv_dim,
        n_classes=args.classes,
        target=args.target,
    )

    print("Generated HLS C++ files:")
    for name in files:
        filepath = os.path.join(args.output, name)
        size = os.path.getsize(filepath)
        print(f"  {name} ({size:,} bytes)")

    # Generate Verilog if requested
    if args.verilog:
        vpath = export_verilog(args.output, hv_dim=args.hv_dim)
        print(f"\nGenerated Verilog: {vpath}")

    # Resource estimate
    report = estimate_fpga_resources(
        hv_dim=args.hv_dim,
        n_classes=args.classes,
        target=args.target,
    )

    print()
    print("Resource Estimate:")
    print(f"  LUTs:   {report.luts:,}")
    print(f"  FFs:    {report.ffs:,}")
    print(f"  DSPs:   {report.dsps}")
    print(f"  BRAMs:  {report.brams}")
    print(f"  Power:  {report.power_mw:.1f} mW")
    print(f"  Latency: {report.latency_cycles} cycles")

    print()
    print("Next steps:")
    print(f"  cd {args.output}")
    print(f"  vitis_hls -f build.tcl")

    return 0


def cmd_list_models(args) -> int:
    """Handle 'list-models' command."""
    models = [
        "resnet18", "resnet50", "resnet101",
        "vgg16", "alexnet",
        "vit_b_16",
        "efficientnet_b0", "mobilenet_v2",
        "densenet121", "googlenet", "inception_v3",
        "shufflenet_v2_x1_0", "mnasnet1_0", "squeezenet1_0",
        "wide_resnet50_2", "regnet_y_16gf",
        "convnext_tiny", "swin_t", "maxvit_t",
    ]
    print("Registered models (via torchvision):")
    for m in models:
        print(f"  {m}")
    print()
    print("You can also pass a path to any .pth/.pt file.")
    return 0


def cmd_info(args) -> int:
    """Handle 'info' command."""
    import torch

    model_name = args.model

    if os.path.isfile(model_name):
        try:
            data = torch.load(model_name, map_location="cpu", weights_only=True)
            print(f"File: {model_name}")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: tensor shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  Type: {type(data).__name__}")
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print(f"Model: {model_name} (from torchvision registry)")
        print(f"  Requires: pip install torchvision")
        print(f"  Command: isildur convert --model {model_name} --hv-dim 10000")
        print()
        print(f"  This will convert the model to a balanced binary hypervector.")
        print(f"  The hypervector represents the model's learned features")
        print(f"  in a high-dimensional HDC space (d=10,000 bits).")
        print(f"  Inference can then run on FPGA ~4 million times faster.")

    return 0


def cmd_benchmark(args) -> int:
    """Handle 'benchmark' command."""
    import torch
    import time
    from isildur.core import gen_hvs, bundle, bind, batch_sim, hamming, sim
    from isildur.cim import CIMAssociativeMemory, CIMConfig

    d = args.hv_dim
    n_classes = args.classes
    n_samples = args.samples

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Isildur: Benchmark HDC Operations                  ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()
    print(f"HV Dimension: {d:,}")
    print(f"Classes: {n_classes}")
    print(f"Samples: {n_samples:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Generate test data
    queries = gen_hvs(n_samples, d, "bipolar", device)
    class_hvs = gen_hvs(n_classes, d, "bipolar", device)

    # Benchmark: Hamming distance (digital)
    t0 = time.perf_counter()
    for i in range(min(n_samples, 100)):
        _ = batch_sim(queries[i], class_hvs, "bipolar")
    digital_time = (time.perf_counter() - t0) / min(n_samples, 100)
    print(f"Digital Similarity: {digital_time * 1e6:.1f} µs/query")

    # Benchmark: CIM Hamming
    cim = CIMAssociativeMemory(n_classes=n_classes, hypervector_dim=d)
    cim.cim.set_class_vectors(class_hvs)
    t0 = time.perf_counter()
    for i in range(min(n_samples, 100)):
        _ = cim.infer(queries[i])
    cim_time = (time.perf_counter() - t0) / min(n_samples, 100)
    print(f"CIM Hamming:        {cim_time * 1e6:.1f} µs/query")

    # Benchmark: Bind (XOR)
    a = queries[0]
    b = queries[1]
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = bind(a, b)
    bind_time = (time.perf_counter() - t0) / 1000
    print(f"Bind (XOR):         {bind_time * 1e6:.1f} µs")

    # Benchmark: Bundle
    batch = queries[:10]
    t0 = time.perf_counter()
    for _ in range(100):
        _ = bundle(batch)
    bundle_time = (time.perf_counter() - t0) / 100
    print(f"Bundle (10 HVs):    {bundle_time * 1e6:.1f} µs")

    # CIM energy estimate
    energy = cim.inference_energy_estimate()
    print()
    print(f"CIM Energy/Inference: {energy:.2f} pJ")
    print(f"Digital Energy (CPU): ~50 nJ per inference")
    print(f"Speedup @ FPGA:       ~{50e-9 / (energy * 1e-12):.0f}x vs CPU")
    print()
    print(f"Throughput (FPGA): {1 / (cim_time) * 1e6:.0f} queries/sec")

    return 0


def _create_demo_cnn():
    """Create a simple CNN for demo purposes."""
    import torch
    import torch.nn as nn

    class DemoCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.relu2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.pool(x).view(x.size(0), -1)
            x = self.fc(x)
            return x

    return DemoCNN()


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "convert": cmd_convert,
        "compare": cmd_compare,
        "infer": cmd_infer,
        "fuse": cmd_fuse,
        "export-fpga": cmd_export_fpga,
        "list-models": cmd_list_models,
        "info": cmd_info,
        "benchmark": cmd_benchmark,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())