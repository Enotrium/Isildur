#!/usr/bin/env python3
"""
benchmark_mnist.py — Isildur MNIST/Fashion-MNIST HDC Classification Benchmark.

Validates Isildur's HDC classification against Serva paper claims:
  - Fashion-MNIST: 88.39% single-epoch, 150J energy
  - MNIST: 96.48% single-epoch, 154J energy

Uses Arthedain encoding (HDC holographic encoding via bind/bundle/permute)
followed by CIM Associative Memory for one-shot classification.

No backpropagation. No training epochs. Just encode and classify.

Requires: pip install torchvision
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isildur.core import gen_hvs, bundle, thresh, ensure_balance, batch_sim
from isildur.cim import CIMAssociativeMemory
from isildur.arthedain import ArthedainEncoder, ARTHEDAIN_DEFAULT_DIM


def load_dataset(name: str = "mnist", batch_size: int = 1000):
    """Load MNIST or Fashion-MNIST."""
    if name == "mnist":
        dataset = torchvision.datasets.MNIST
    else:
        dataset = torchvision.datasets.FashionMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_set = dataset(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = dataset(
        root="./data", train=False, download=True, transform=transform
    )

    return train_set, test_set


def encode_dataset(
    encoder: ArthedainEncoder,
    dataset: torchvision.datasets.VisionDataset,
    max_samples: int = None,
) -> tuple:
    """
    Encode an entire dataset into hypervectors.

    Returns:
        hvs: (n, hv_dim) tensor of encoded hypervectors
        labels: (n,) tensor of labels
    """
    hvs = []
    labels = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in range(n):
        img, label = dataset[i]
        # Encode the image tensor
        hv = encoder.encode_image(img)
        hvs.append(hv)
        labels.append(label)

    return torch.stack(hvs), torch.tensor(labels)


def classify_one_shot(
    train_hvs: torch.Tensor,
    train_labels: torch.Tensor,
    test_hvs: torch.Tensor,
    n_classes: int = 10,
) -> float:
    """
    One-shot HDC classification.

    1. Bundle all training HVs per class → class prototypes
    2. Classify test HVs by minimum Hamming distance to prototypes
    3. Return accuracy
    """
    # Build class prototypes via bundling
    class_prototypes = torch.zeros(n_classes, train_hvs.shape[1])

    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            class_prototypes[c] = bundle(train_hvs[mask])

    class_prototypes = thresh(class_prototypes)

    # Classify test set
    correct = 0
    total = test_hvs.shape[0]

    for i in range(total):
        query = test_hvs[i]
        similarities = batch_sim(query, class_prototypes, "bipolar")
        pred = int(similarities.argmax().item())
        if pred == test_labels[i].item():
            correct += 1

    return correct / total


def benchmark_dataset(name: str = "mnist", hv_dim: int = 8192, max_train: int = 5000):
    """Run full benchmark on one dataset."""
    print(f"\n{'='*60}")
    print(f"Isildur HDC Benchmark: {name.upper()}")
    print(f"{'='*60}")
    print(f"HV Dimension: {hv_dim}")
    print(f"Max training samples: {max_train}")
    print()

    # Load data
    train_set, test_set = load_dataset(name)
    print(f"Train set: {len(train_set)} samples")
    print(f"Test set:  {len(test_set)} samples")

    # Setup encoder
    encoder = ArthedainEncoder(hv_dim=hv_dim, seed=42)

    # Encode training set
    t0 = time.perf_counter()
    train_hvs, train_labels = encode_dataset(encoder, train_set, max_train)
    encode_time = time.perf_counter() - t0
    print(f"Encoding: {encode_time:.1f}s for {train_hvs.shape[0]} samples")

    # Encode test set
    t0 = time.perf_counter()
    test_hvs, test_labels = encode_dataset(encoder, test_set, max_samples=2000)
    test_encode_time = time.perf_counter() - t0

    # Classify (one-shot)
    t0 = time.perf_counter()
    accuracy = classify_one_shot(train_hvs, train_labels, test_hvs)
    classify_time = time.perf_counter() - t0
    total_time = encode_time + test_encode_time + classify_time

    # CIM energy estimate
    cim = CIMAssociativeMemory(n_classes=10, hypervector_dim=hv_dim)
    energy_per_inference = cim.inference_energy_estimate()  # pJ
    total_energy_pj = energy_per_inference * test_hvs.shape[0]

    # Results
    print(f"\n{'─'*40}")
    print(f"Results: {name.upper()}")
    print(f"{'─'*40}")
    print(f"Accuracy:     {accuracy*100:.2f}%")
    print(f"Total time:   {total_time:.1f}s")
    print(f"Encode time:  {encode_time + test_encode_time:.1f}s")
    print(f"Classify:     {classify_time:.1f}s ({test_hvs.shape[0]} queries)")
    print(f"CIM energy:   {total_energy_pj:.0f} pJ (~{total_energy_pj/1e9:.3f} mJ)")
    print(f"Per query:    {energy_per_inference:.1f} pJ")

    # Comparison with Serva paper
    paper_acc = {"mnist": 96.48, "fashion_mnist": 88.39}.get(name, None)
    if paper_acc:
        delta = accuracy * 100 - paper_acc
        print(f"Paper acc:    {paper_acc}% (Δ={delta:+.2f}%)")

    print()

    return accuracy, total_time


def main():
    """Run benchmarks on both datasets."""
    print("Isildur HDC Classification Benchmark")
    print("═══════════════════════════════════")
    print("Validates single-epoch HDC classification")
    print("against Serva/Arthedain Standard claims.")
    print()

    results = {}

    # MNIST
    acc, t = benchmark_dataset("mnist", hv_dim=4096, max_train=5000)
    results["mnist"] = {"accuracy": acc, "time": t}

    # Fashion-MNIST
    acc2, t2 = benchmark_dataset("fashion_mnist", hv_dim=4096, max_train=5000)
    results["fashion_mnist"] = {"accuracy": acc2, "time": t2}

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, r in results.items():
        paper_acc = {"mnist": 96.48, "fashion_mnist": 88.39}[name]
        delta = r["accuracy"] * 100 - paper_acc
        status = "✓ MATCHES" if abs(delta) < 5 else "△ CLOSE" if abs(delta) < 10 else "✗ OFF"
        print(f"  {name}: {r['accuracy']*100:.1f}% "
              f"(paper: {paper_acc}%, Δ={delta:+.1f}%) — {status} | {r['time']:.1f}s")


if __name__ == "__main__":
    main()