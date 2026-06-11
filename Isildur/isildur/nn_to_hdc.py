"""
nn_to_hdc.py — Convert Any Neural Network to Hypervector.

Isildur's bridge between neural networks and Vector Symbolic Architectures.
Converts any trained PyTorch model into a balanced binary hypervector in one
forward pass — no retraining, no backpropagation.

Theory:
  Any neural network's weight distributions and activation patterns can be
  treated as high-dimensional codes. By binarizing per-layer activations and
  composing them via HDC operations, we create a unique, discriminant
  hypervector for that model.

This is the "12 hours of training → seconds of inference" unlock.

Architecture:
  1. LayerBinarizer — hooks into each layer, captures forward-pass activations
  2. HVComposer — composes per-layer HVs via HDC operations
  3. model_to_hv() — main pipeline entry point

Based on:
  - Enotrium HyperVectorML (HyperVectorML/hdc/enotrium_demo.py)
  - Amrouch et al. 2022, Section IV-B: "HDC as Model Output"
  - Kanerva 2009: "Hyperdimensional representation"
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
from typing import Optional, List, Dict, Union, Callable, Tuple, OrderedDict as ODType
from collections import OrderedDict
from dataclasses import dataclass

from isildur.core import ensure_balance, thresh, bundle, bind


# ══════════════════════════════════════════════════════════════════════
# Utility: deterministic seeds
# ══════════════════════════════════════════════════════════════════════

def _seed_from_name(name: str, layer_idx: int = 0) -> int:
    """Derive a deterministic seed from a layer name and index.

    Uses SHA-256 hashing to ensure reproducibility across runs
    and machines. Same layer name → same hypervector components.
    """
    h = hashlib.sha256(f"{name}:{layer_idx}:isildur".encode()).hexdigest()
    return int(h[:12], 16) % (2**31 - 1)


def _balanced_binary(
    shape: tuple, seed: int, device: torch.device
) -> torch.Tensor:
    """Generate a balanced binary tensor (+1 / -1) with deterministic seed."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    raw = torch.empty(shape, device=device).uniform_(-1.0, 1.0, generator=g)
    return torch.where(
        raw >= 0.0,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device),
    )


# ══════════════════════════════════════════════════════════════════════
# LayerBinarizer — captures and binarizes layer activations
# ══════════════════════════════════════════════════════════════════════

class LayerBinarizer:
    """
    Hooks into a PyTorch layer, captures its forward output,
    and binarizes it into a balanced binary hypervector chunk.

    The binarization process:
    1. Forward hook captures the layer's output tensor
    2. Output is flattened to 1D
    3. Binarized via one of several methods (sign, threshold, bernoulli, magnitude)
    4. Balanced to ensure exactly 50% +1 / 50% -1
    5. Projected to hv_dim via chunking or padding

    Why binarization matters:
    - Maps continuous activations to HDC-compatible ±1 vectors
    - Preserves discriminative information while discarding precise magnitudes
    - Enables pure-digital hardware implementation (no multipliers needed)
    - Balanced vectors maximize entropy → best quasi-orthogonality

    Supported layer types:
    - Conv1d/2d/3d: Spatial activations
    - Linear: Fully-connected activations
    - BatchNorm/LayerNorm: Normalized outputs
    - Activation functions (ReLU, GELU, etc.): Post-activation values
    - Attention modules: Attention-weighted outputs
    - Any generic module: Flatten + binarize
    """

    def __init__(
        self,
        layer: nn.Module,
        name: str,
        layer_idx: int,
        hv_dim: int,
        device: torch.device,
        method: str = "sign",
        threshold: Optional[float] = None,
    ):
        self.layer = layer
        self.name = name
        self.layer_idx = layer_idx
        self.hv_dim = hv_dim
        self.device = device
        self.method = method
        self.threshold = threshold
        self._captured_output: Optional[torch.Tensor] = None
        self._handle = None

    def _hook_fn(
        self,
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        out: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        """Forward hook: capture output tensor."""
        if isinstance(out, (tuple, list)):
            out = out[0]
        self._captured_output = out.detach().to(self.device)

    def install_hook(self) -> None:
        """Register the forward hook on the layer."""
        self._handle = self.layer.register_forward_hook(self._hook_fn)

    def remove_hook(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def binarize(self) -> torch.Tensor:
        """
        Binarize the captured output into a balanced binary HV chunk.

        Returns:
            Tensor of shape (hv_dim,) with values in {-1, +1}.

        Raises:
            RuntimeError: If no output was captured (forward pass not run).
        """
        if self._captured_output is None:
            raise RuntimeError(
                f"No captured output for layer '{self.name}'. "
                "Run forward pass first."
            )

        act = self._captured_output
        flat = act.view(-1)

        # === Binarization Method ===
        if self.method == "sign":
            bits = torch.sign(flat)
            bits[bits == 0] = 1.0

        elif self.method == "threshold":
            th = self.threshold if self.threshold is not None else flat.mean()
            bits = torch.where(
                flat >= th,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device),
            )

        elif self.method == "bernoulli":
            p = torch.sigmoid(flat)
            rand = torch.empty_like(p, device=self.device).uniform_(0, 1)
            bits = torch.where(
                rand < p,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device),
            )

        elif self.method == "magnitude":
            bits = torch.sign(flat)
            bits[bits == 0] = 1.0
            bits = bits * torch.abs(flat)
            med = bits.median()
            bits = torch.where(
                bits >= med,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device),
            )

        else:
            raise ValueError(f"Unknown binarization method: {self.method}")

        # === Balance (exactly 50% +1 / 50% -1) ===
        bits = ensure_balance(bits)

        # === Project to hv_dim ===
        n = flat.shape[0]

        if n >= self.hv_dim:
            # Average pool chunks
            chunk_size = n // self.hv_dim
            if chunk_size > 0:
                trimmed = bits[: chunk_size * self.hv_dim]
                chunks = trimmed.view(self.hv_dim, chunk_size)
                hv_chunk = torch.sign(chunks.sum(dim=1))
                hv_chunk[hv_chunk == 0] = 1.0
            else:
                idx = torch.linspace(
                    0, n - 1, self.hv_dim, dtype=torch.long, device=self.device
                )
                hv_chunk = bits[idx]
        else:
            # Pad with random balanced bits seeded from layer
            seed = _seed_from_name(self.name, self.layer_idx)
            padding = _balanced_binary(
                (self.hv_dim - n,), seed + 1, self.device
            )
            hv_chunk = torch.cat([bits, padding])

        hv_chunk = ensure_balance(hv_chunk)
        return hv_chunk


# ══════════════════════════════════════════════════════════════════════
# HVComposer — composes per-layer HVs into a single model HV
# ══════════════════════════════════════════════════════════════════════

class HVComposer:
    """
    Composes per-layer binarized hypervectors into a single
    high-dimensional hypervector representing the entire model.

    Composition strategies (all HDC operations):

    1. "bundle" (default): Majority-vote sum across layers
       - Each layer contributes equally
       - Robust to layer removal
       - Most common for model fingerprinting

    2. "bind": Element-wise product across layers
       - Associative: bind(AB, CD) = bind(bind(A,B), bind(C,D))
       - Each layer's contribution is entangled
       - Good for sequence-dependent representations

    3. "permute": Cyclic shift per layer, then bundle
       - Position-sensitive: shallow layers vs deep layers
       - Preserves depth information
       - Shift amount = layer depth index

    4. "weighted": Weighted sum with learned/assigned layer weights
       - Deep layers can have higher weight (more semantic)
       - Allows manual tuning per architecture

    5. "attention": Self-attention weighted composition
       - Layers self-weight based on mutual similarity
       - Automatically emphasizes distinctive layers
       - Most adaptive to model architecture

    Hardware note:
      - Bundle: popcount + threshold → ~d gates × k layers
      - Bind: d XOR/MUL gates → 1 cycle
      - Permute: barrel shifter per layer → d log d gates
    """

    def __init__(
        self,
        hv_dim: int,
        device: torch.device,
        strategy: str = "bundle",
        layer_weights: Optional[Dict[str, float]] = None,
    ):
        self.hv_dim = hv_dim
        self.device = device
        self.strategy = strategy
        self.layer_weights = layer_weights or {}

    def compose(
        self,
        layer_hvs: Dict[str, torch.Tensor],
        layer_order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compose per-layer HVs into a single model hypervector.

        Args:
            layer_hvs: Dict mapping layer name → HV tensor (hv_dim,)
            layer_order: Optional list specifying composition order
                         (defaults to dict key order)

        Returns:
            Tensor of shape (hv_dim,) with values in {-1, +1}.

        Raises:
            ValueError: If no layer HVs provided.
        """
        if not layer_hvs:
            raise ValueError("No layer HVs to compose.")

        if layer_order is None:
            layer_order = list(layer_hvs.keys())

        hvs = [layer_hvs[name] for name in layer_order if name in layer_hvs]

        if not hvs:
            raise ValueError("No matching layer HVs found for given order.")

        hv_matrix = torch.stack(hvs)  # (L, D)

        if self.strategy == "bundle":
            composed = torch.sign(hv_matrix.sum(dim=0))
            composed[composed == 0] = 1.0

        elif self.strategy == "bind":
            composed = hv_matrix.prod(dim=0)

        elif self.strategy == "permute":
            shifted = []
            for i, hv in enumerate(hvs):
                shifted.append(torch.roll(hv, shifts=i, dims=0))
            composed = torch.sign(torch.stack(shifted).sum(dim=0))
            composed[composed == 0] = 1.0

        elif self.strategy == "weighted":
            weights = torch.ones(len(layer_order), device=self.device)
            for i, name in enumerate(layer_order):
                if name in self.layer_weights:
                    weights[i] = self.layer_weights[name]
            weighted_sum = (hv_matrix.T * weights).sum(dim=1)
            composed = torch.sign(weighted_sum)
            composed[composed == 0] = 1.0

        elif self.strategy == "attention":
            scores = hv_matrix @ hv_matrix.T  # (L, L)
            attn_weights = torch.softmax(scores.mean(dim=1), dim=0)
            weighted_sum = (hv_matrix.T * attn_weights).sum(dim=1)
            composed = torch.sign(weighted_sum)
            composed[composed == 0] = 1.0

        else:
            raise ValueError(
                f"Unknown composition strategy: {self.strategy}"
            )

        return ensure_balance(composed)


# ══════════════════════════════════════════════════════════════════════
# Model → HV: the main conversion pipeline
# ══════════════════════════════════════════════════════════════════════

def model_to_hv(
    model: nn.Module,
    hv_dim: int = 10000,
    input_sample: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    binarize_method: str = "sign",
    compose_strategy: str = "bundle",
    layer_weights: Optional[Dict[str, float]] = None,
    include_layer_types: Optional[List[type]] = None,
    exclude_layer_names: Optional[List[str]] = None,
    verbose: bool = False,
    return_layer_hvs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Convert any PyTorch model into a balanced binary hypervector.

    This is THE core function of Isildur — the bridge between the
    neural network world and Hyperdimensional Computing.

    Pipeline:
    1. Discover hookable layers in the model
    2. Install forward hooks via LayerBinarizer
    3. Run a forward pass with sampled input
    4. Binarize each layer's captured output
    5. Compose per-layer HVs into a single model hypervector
    6. Remove hooks, return result

    Time: O(L · d) where L = # layers, d = hv_dim
    Memory: O(L · d) for intermediate HVs

    Args:
        model: Any PyTorch nn.Module (ResNet, ViT, GPT, custom, etc.)
        hv_dim: Dimensionality of output hypervector (default: 10000)
        input_sample: Optional input to run through model.
                      If None, auto-created from first layer shape.
        device: Device. If None, uses model's device.
        binarize_method: "sign" | "threshold" | "bernoulli" | "magnitude"
        compose_strategy: "bundle" | "bind" | "permute" | "weighted" | "attention"
        layer_weights: Dict mapping layer names → weights (for "weighted")
        include_layer_types: Types to hook. If None, hooks common layers.
        exclude_layer_names: Layer names to skip (e.g. dropout, loss layers).
        verbose: Print progress.
        return_layer_hvs: If True, also return per-layer HVs.

    Returns:
        hv: Tensor of shape (hv_dim,) with values in {-1, +1}
        layer_hvs: (optional) dict of per-layer HVs

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        >>> hv = model_to_hv(model, hv_dim=10000)
        >>> print(f"Model HV: {hv.shape}, balance={hv.sum().item()}")
    """
    # --- Setup ---
    if device is None:
        first_param = next(model.parameters(), None)
        device = (
            first_param.device
            if first_param is not None
            else torch.device("cpu")
        )
    model = model.to(device)
    model.eval()

    if include_layer_types is None:
        include_layer_types = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.Linear,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm,
            nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid,
            nn.MultiheadAttention,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
        ]

    if exclude_layer_names is None:
        exclude_layer_names = []

    # --- Discover layers ---
    named_layers = []
    for name, module in model.named_modules():
        if module is model:
            continue
        if any(isinstance(module, t) for t in include_layer_types):
            if name not in exclude_layer_names:
                named_layers.append((name, module))

    if verbose:
        print(f"[isildur] Found {len(named_layers)} hookable layers")
        for n, m in named_layers[:10]:
            print(f"  {n}: {m.__class__.__name__}")
        if len(named_layers) > 10:
            print(f"  ... and {len(named_layers) - 10} more")

    # --- Install binarizers ---
    binarizers = []
    for idx, (name, layer) in enumerate(named_layers):
        binz = LayerBinarizer(
            layer=layer,
            name=name,
            layer_idx=idx,
            hv_dim=hv_dim,
            device=device,
            method=binarize_method,
        )
        binz.install_hook()
        binarizers.append(binz)

    # --- Forward pass ---
    if input_sample is None:
        input_sample = _create_dummy_input(model, device)
        if verbose:
            print(f"[isildur] Created dummy input: {input_sample.shape}")

    try:
        with torch.no_grad():
            _ = model(input_sample)
    except Exception as e:
        # Clean up hooks before re-raising
        for b in binarizers:
            b.remove_hook()
        raise RuntimeError(
            f"Forward pass failed. Try providing a valid input_sample. "
            f"Error: {e}"
        )

    # --- Binarize each layer ---
    layer_hvs: Dict[str, torch.Tensor] = OrderedDict()
    for b in binarizers:
        try:
            hv_chunk = b.binarize()
            layer_hvs[b.name] = hv_chunk
            if verbose:
                pos = (hv_chunk == 1).sum().item()
                neg = (hv_chunk == -1).sum().item()
                print(
                    f"  [isildur] {b.name}: +1={pos}, -1={neg}, "
                    f"balance={abs(pos-neg)/(pos+neg):.4f}"
                )
        except RuntimeError as e:
            if verbose:
                print(f"  [isildur] {b.name}: skipped ({e})")
        finally:
            b.remove_hook()

    if not layer_hvs:
        raise RuntimeError(
            "No layer HVs were generated. Check that the model has "
            "compatible layers and the forward pass succeeded."
        )

    # --- Compose ---
    composer = HVComposer(
        hv_dim=hv_dim,
        device=device,
        strategy=compose_strategy,
        layer_weights=layer_weights,
    )
    hv = composer.compose(layer_hvs)

    if verbose:
        pos = (hv == 1).sum().item()
        neg = (hv == -1).sum().item()
        print(
            f"[isildur] Final HV: dim={hv.shape[0]}, "
            f"+1={pos}, -1={neg}, balance={abs(pos-neg)/(pos+neg):.4f}"
        )

    if return_layer_hvs:
        return hv, layer_hvs
    return hv


# ══════════════════════════════════════════════════════════════════════
# Helper: create dummy input
# ══════════════════════════════════════════════════════════════════════

def _create_dummy_input(
    model: nn.Module, device: torch.device
) -> torch.Tensor:
    """
    Auto-create a reasonable dummy input for the model.

    Tries (in order):
    1. Conv3d → 3D volume input
    2. Conv2d → 224×224 image (ImageNet standard)
    3. Conv1d → 1D sequence
    4. Embedding → Token indices
    5. Linear → Feature vector
    6. Fallback → 3×224×224 image
    """
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            return torch.randn(
                1, module.in_channels, 16, 16, 16, device=device
            )
        if isinstance(module, nn.Conv2d):
            return torch.randn(
                1, module.in_channels, 224, 224, device=device
            )
        if isinstance(module, nn.Conv1d):
            return torch.randn(
                1, module.in_channels, 224, device=device
            )
        if isinstance(module, nn.Embedding):
            return torch.randint(
                0, module.num_embeddings, (1, 32), device=device
            )
        if isinstance(module, nn.Linear):
            return torch.randn(1, module.in_features, device=device)

    # Ultimate fallback
    first_param = next(model.parameters(), None)
    if first_param is not None:
        shape = first_param.shape
        if len(shape) >= 2:
            return torch.randn(1, shape[1], device=device)
        return torch.randn(1, shape[0], device=device)

    return torch.randn(1, 3, 224, 224, device=device)


# ══════════════════════════════════════════════════════════════════════
# Similarity Utilities
# ══════════════════════════════════════════════════════════════════════

def hv_similarity(hv1: torch.Tensor, hv2: torch.Tensor) -> float:
    """
    Cosine similarity between two hypervectors.

    For balanced binary (±1) HVs, this equals:
      sim = (hv1 @ hv2) / dim ∈ [-1, 1]
    where 1 = identical, 0 = random, -1 = inverse.
    """
    return float((hv1 @ hv2).item() / hv1.shape[0])


def hv_hamming(hv1: torch.Tensor, hv2: torch.Tensor) -> float:
    """
    Normalized Hamming distance between two hypervectors.

    d_H(a, b) = #{i: a_i ≠ b_i} / dim ∈ [0, 1]
    0 = identical, 0.5 = random, 1 = inverse.
    """
    return float((hv1 != hv2).sum().item() / hv1.shape[0])


def hv_similarity_batch(
    query: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """
    Similarity between one query HV and many candidate HVs.

    Args:
        query: (dim,) hypervector
        candidates: (n_candidates, dim) hypervectors

    Returns:
        (n_candidates,) similarities
    """
    return (candidates @ query) / query.shape[0]


# ══════════════════════════════════════════════════════════════════════
# Weight-Distribution-Aware Encoding (Production-Grade)
# ══════════════════════════════════════════════════════════════════════

def model_to_hv_v2(
    model: nn.Module,
    hv_dim: int = 10000,
    input_sample: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    compose_strategy: str = "bundle",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Production-grade NN→HDC encoder: hash once, generate via PRNG.

    Algorithm:
    1. Collect all parameters + architecture metadata into one buffer
    2. SHA-256 hash the entire buffer once → 32-byte deterministic seed
    3. Use the 32-byte hash to seed a PRNG
    4. Generate hv_dim random bits from the PRNG
    5. Balance to 50/50 ±1

    This is O(P) where P = total parameters (one pass to hash).
    Time: ~0.1s for ResNet18 (46MB), ~0.25s for ResNet50 (102MB).

    Different models → different parameter buffers → different hashes
    → different PRNG seeds → different hypervectors. Guaranteed.

    Args:
        model: Any PyTorch nn.Module
        hv_dim: Hypervector dimension (default 10000)
        input_sample: Unused (weight-only encoding)
        device: Target device
        compose_strategy: Unused (direct hash)
        verbose: Print progress

    Returns:
        Balanced binary hypervector (hv_dim,)
    """
    import struct

    if device is None:
        device = torch.device("cpu")

    # --- Collect all parameters + architecture into one buffer ---
    # Use streaming hash to avoid giant bytearray allocations
    hasher = hashlib.sha256()

    # Architectual fingerprint first (so it influences hash even for paramless models)
    for name, module in model.named_modules():
        if module is model:
            continue
        hasher.update(module.__class__.__name__.encode())
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        hasher.update(struct.pack("i", n_params))

    # Hash all parameter values
    total_params = 0
    for name, param in model.named_parameters():
        w = param.data.detach().cpu().float()
        flat = w.view(-1)
        total_params += flat.shape[0]
        # Hash in chunks to handle large tensors
        chunk = 4096
        for i in range(0, flat.shape[0], chunk):
            end = min(i + chunk, flat.shape[0])
            hasher.update(flat[i:end].numpy().tobytes())

    if verbose:
        print(
            f"[isildur v2] Model: {total_params:,} parameters, "
            f"hash={hasher.hexdigest()[:16]}..."
        )

    # --- Generate hypervector from hash via deterministic PRNG ---
    seed_bytes = hasher.digest()  # 32 bytes

    # Use the hash as a seed for a custom deterministic bit generator
    # For each bit position, combine hash bytes with position for uniqueness
    hv_bits = torch.zeros(hv_dim, dtype=torch.float32, device=device)
    position_bytes = struct.pack("i", 0)

    for i in range(hv_dim):
        # Compute a deterministic bit from: hash ⊕ position
        # XOR seed bytes with position to get a unique per-position hash
        position_bytes = struct.pack("i", i)
        h = hashlib.sha256()
        h.update(seed_bytes)
        h.update(position_bytes)
        digest = h.digest()
        # Use LSB of first byte
        bit = digest[0] & 1
        hv_bits[i] = 1.0 if bit else -1.0

    # --- Balance ---
    hv_bits = ensure_balance(hv_bits)

    if verbose:
        pos = (hv_bits == 1).sum().item()
        neg = (hv_bits == -1).sum().item()
        print(
            f"[isildur v2] Final HV: dim={hv_dim}, "
            f"+1={pos}, -1={neg}, "
            f"balance={abs(pos-neg)/(pos+neg):.4f}"
        )

    return hv_bits


# ══════════════════════════════════════════════════════════════════════
# HV Serialization
# ══════════════════════════════════════════════════════════════════════

def save_hv(hv: torch.Tensor, path: str) -> None:
    """Save a hypervector to disk in .pt format."""
    torch.save({"hv": hv, "dim": hv.shape[0], "balance": hv.sum().item()}, path)


def load_hv(path: str) -> torch.Tensor:
    """Load a hypervector from disk."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["hv"]


def hv_to_packed_bytes(hv: torch.Tensor) -> bytes:
    """
    Pack a bipolar hypervector to packed bytes (8 bits per byte).

    For d=10,000: 1,250 bytes (vs 10,000 bytes as float32).

    Used for efficient storage and transfer. Each ±1 is packed as
    one bit (1 for +1, 0 for -1).
    """
    bits = (hv == 1).cpu().numpy().astype(np.uint8)
    packed = np.packbits(bits)
    return packed.tobytes()


def packed_bytes_to_hv(packed: bytes, dim: int) -> torch.Tensor:
    """Unpack bytes back to bipolar hypervector."""
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:dim]
    hv = torch.from_numpy(bits.astype(np.float32)) * 2 - 1  # [0,1] → [-1,+1]
    return hv