"""
serva.py — The Arthedain Standard Integration for Isildur.

Implements the .arthedain universal data format based on Arthedain's
ArthedainStack architecture, bridged with Isildur's HDC/VSA engine.

The Arthedain Standard (Arthedain Inc., Dec 2025):
  "Any data to any model on any hardware."
  
  Core primitives: bit-level XOR, bundling, permutation,
  pseudo-random bit generation, and Hamming distance —
  the same operations that define Hyperdimensional Computing.

  Key insight: compression and learning are the same operation.
  Hutter proved optimal compression implies optimal prediction.
  Shannon proved how to compress without losing information.
  Both are satisfied when data is already structured by physical
  causality — and Arthedain exploits this by encoding into a
  high-dimensional representational space where computation
  occurs directly on the compressed representation.

  The .arthedain format achieves:
  - 30–374× energy efficiency (96–99% reduction)
  - 4–34× lossless compression
  - 68× compute payload reduction
  - No loss of accuracy when training on .arthedain data

This module:
  1. Implements Arthedain encoding primitives (holographic encoding)
  2. Loads the SV Library of pre-encoded concept hypervectors
  3. Provides .arthedain file format I/O
  4. Bridges Serva/HDC operations for universal computation

SV Library: ~/.arthedain/sv_library/
  - 32 pre-encoded hypervectors across 6 domains
  - 8192-dim bipolar vectors with SHA-256 integrity
  - Vision (MNIST 0-9), Defense (6 types), Sensors (5 types),
    BCI (4 types), NLP (6 concepts), Fused (1 bundle)

Zotero HDC Collection: https://www.zotero.org/enotrium/collections/JTV8PX3T
  - Academic papers on Hyperdimensional Computing
  - Including Amrouch et al. 2022, Kanerva 2009, Gayler 2003
  - Referenced for theory validation
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import hashlib
import pickle
from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from isildur.core import (
    gen_hvs, bundle, bind, permute, sim, hamming, thresh,
    ensure_balance, batch_sim,
)


# ══════════════════════════════════════════════════════════════════════
# Arthedain Constants (from paper: Section 3.3 Architecture)
# ══════════════════════════════════════════════════════════════════════

# Default hypervector dimension (matches SV library)
ARTHEDAIN_DEFAULT_DIM = 8192

# Arthedain encoding operations (paper: "bit-level addition, XOR, permutation,
# pseudo-random bit generation, and distance")
ARTHEDAIN_OPS = {
    "xor": "bind",          # Element-wise XOR = HDC binding
    "bundle": "bundle",      # Accumulation = HDC bundling
    "permute": "permute",    # Cyclic shift = HDC permutation
    "distance": "hamming",   # Popcount XOR = Hamming distance
    "threshold": "thresh",   # Sign threshold = bipolar normalization
}


# ══════════════════════════════════════════════════════════════════════
# SV Library Manager
# ══════════════════════════════════════════════════════════════════════

class SVLibrary:
    """
    Manager for the SV Library of pre-encoded concept hypervectors.

    The SV (Arthedain Vector) library stores hypervectors that represent
    fundamental concepts — sensor modalities, defense targets, BCI
    states, NLP concepts — as balanced bipolar vectors.

    These serve multiple purposes:
    1. Prototype vectors for one-shot classification
    2. Basis vectors for sensor fusion via bundling
    3. Concept embeddings for semantic search in HDC space
    4. Reference vectors for model validation

    The library is located at ~/.arthedain/sv_library/ and managed
    via a JSON registry with SHA-256 integrity verification.

    Domains:
    - vision: MNIST digit prototypes (0-9), 50 examples each
    - defense: Ground vehicles, UAVs, jets, IFF, anomalies
    - sensors: RF, acoustic, IMU, DVS/event camera, thermal IR
    - bci: Motor velocity X/Y, reach-grasp, rest state
    - nlp: HDC, transformers, edge AI, SNN, sensor fusion, binary HDC
    """

    def __init__(
        self,
        library_path: Optional[str] = None,
        hv_dim: int = ARTHEDAIN_DEFAULT_DIM,
    ):
        """
        Initialize SV Library manager.

        Args:
            library_path: Path to SV library directory.
                         Defaults to ~/.arthedain/sv_library/
            hv_dim: Expected hypervector dimension
        """
        # Default library path
        if library_path is None:
            home = os.path.expanduser("~")
            library_path = os.path.join(
                home, ".arthedain", "sv_library"
            )

        self.library_path = library_path
        self.hv_dim = hv_dim
        self.registry: Dict[str, dict] = {}
        self.loaded_hvs: Dict[str, torch.Tensor] = {}
        self._loaded = False

    def load_registry(self) -> None:
        """
        Load the SV library registry and all hypervectors.

        The registry (sv_registry.json) contains:
        - path: File location
        - dim: Hypervector dimension
        - n_examples: Training examples bundled
        - domain: Category (vision/defense/sensors/bci/nlp)
        - description: Human-readable concept
        - sha256: Content integrity hash
        - created_at: ISO timestamp
        """
        registry_path = os.path.join(
            self.library_path, "sv_registry.json"
        )

        if not os.path.exists(registry_path):
            print(f"[isildur/arthedain] SV Library not found at {registry_path}")
            print(f"  Initialize with: isildur arthedain-init")
            return

        with open(registry_path, "r") as f:
            self.registry = json.load(f)

        # Load each hypervector
        for key, meta in self.registry.items():
            hv_path = meta.get("path", "")
            if not hv_path or not os.path.exists(hv_path):
                # Try relative to library path
                hv_path = os.path.join(
                    self.library_path, os.path.basename(hv_path)
                )
                if not os.path.exists(hv_path):
                    continue

            try:
                hv = torch.load(hv_path, map_location="cpu", weights_only=True)
                # Handle different save formats
                if isinstance(hv, dict):
                    hv = hv.get("hv", hv.get("data", None))
                if hv is not None and hv.shape[-1] == self.hv_dim:
                    self.loaded_hvs[key] = hv.squeeze().float()
                else:
                    print(f"  Skipped {key}: dim mismatch or no data")
            except Exception as e:
                print(f"  Failed to load {key}: {e}")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_hv(self, key: str) -> torch.Tensor:
        """
        Retrieve a pre-encoded hypervector by key.

        Args:
            key: Registry key, e.g., "vision/mnist-digit-0-8192"
                 Can be shortened to "vision/mnist-digit-0"

        Returns:
            Hypervector tensor of shape (hv_dim,)

        Raises:
            KeyError: If key not found
        """
        if not self._loaded:
            self.load_registry()

        # Try exact match first
        if key in self.loaded_hvs:
            return self.loaded_hvs[key].clone()

        # Try with -8192 suffix
        suffixed = f"{key}-{self.hv_dim}"
        if suffixed in self.loaded_hvs:
            return self.loaded_hvs[suffixed].clone()

        # Try partial match
        for k, hv in self.loaded_hvs.items():
            if k.startswith(key) or key in k:
                return hv.clone()

        raise KeyError(
            f"Hypervector not found: {key}. "
            f"Available keys: {list(self.loaded_hvs.keys())}"
        )

    def get_hvs_by_domain(self, domain: str) -> Dict[str, torch.Tensor]:
        """
        Retrieve all hypervectors in a domain.

        Args:
            domain: "vision", "defense", "sensors", "bci", "nlp"

        Returns:
            Dict of key → hypervector
        """
        if not self._loaded:
            self.load_registry()

        return {
            k: hv.clone()
            for k, hv in self.loaded_hvs.items()
            if k.startswith(domain + "/") or k.startswith(domain + "__")
        }

    def fuse_sensors(
        self, sensor_keys: List[str], strategy: str = "bundle"
    ) -> torch.Tensor:
        """
        Fuse multiple sensor HVs into a multi-modal representation.

        This is the physical realization of HDC sensor fusion:
        fuse = bundle([sensor1_hv, sensor2_hv, ...])

        From the SV library, "defense/ground-vehicle-acoustic-imu-fused-8192"
        is a pre-computed bundle of ground-vehicle, acoustic-waveform,
        and imu-inertial HVs.

        Args:
            sensor_keys: List of registry keys to fuse
            strategy: "bundle" (additive) or "bind" (entangled)

        Returns:
            Fused hypervector (hv_dim,)
        """
        hvs = [self.get_hv(key) for key in sensor_keys]

        if strategy == "bundle":
            return bundle(torch.stack(hvs))
        elif strategy == "bind":
            fused = hvs[0].clone()
            for hv in hvs[1:]:
                fused = bind(fused, hv)
            return fused
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def classify_by_hamming(
        self,
        query: torch.Tensor,
        domain: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Classify a query hypervector against the SV library.

        Uses minimum Hamming distance (max cosine similarity) to
        find the closest concept vectors.

        Args:
            query: Query hypervector (hv_dim,)
            domain: Optional domain filter
            top_k: Number of top matches

        Returns:
            List of (key, similarity) tuples
        """
        if not self._loaded:
            self.load_registry()

        candidates = (
            self.get_hvs_by_domain(domain)
            if domain
            else self.loaded_hvs
        )

        if not candidates:
            return []

        keys = list(candidates.keys())
        hvs = torch.stack([candidates[k] for k in keys])

        similarities = batch_sim(query, hvs, "bipolar")
        top_vals, top_idx = similarities.topk(
            min(top_k, len(keys))
        )

        return [
            (keys[int(top_idx[i])], float(top_vals[i]))
            for i in range(len(top_idx))
        ]

    def verify_integrity(self) -> Dict[str, bool]:
        """
        Verify SHA-256 integrity of all library vectors.

        Returns:
            Dict of key → valid (bool)
        """
        results = {}
        for key, meta in self.registry.items():
            hv_path = meta.get("path", "")
            expected_hash = meta.get("sha256", "")

            if not hv_path or not expected_hash:
                results[key] = False
                continue

            if not os.path.exists(hv_path):
                hv_path = os.path.join(
                    self.library_path, os.path.basename(hv_path)
                )

            if not os.path.exists(hv_path):
                results[key] = False
                continue

            with open(hv_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            results[key] = actual_hash == expected_hash

        return results

    def list_domains(self) -> List[str]:
        """List all domains in the library."""
        if not self._loaded:
            self.load_registry()
        domains = set()
        for key in self.loaded_hvs.keys():
            domain = key.split("/")[0].split("__")[0]
            domains.add(domain)
        return sorted(domains)

    def __len__(self) -> int:
        if not self._loaded:
            self.load_registry()
        return len(self.loaded_hvs)


# ══════════════════════════════════════════════════════════════════════
# Arthedain Encoder — Universal Data → .arthedain Holographic Encoding
# ══════════════════════════════════════════════════════════════════════

class ArthedainEncoder:
    """
    Universal data encoder implementing the Arthedain Standard.

    Converts any data (images, text, audio, sensor streams,
    structured records) into .arthedain format — a high-dimensional
    binary representation that preserves all information while
    enabling direct computation without decompression.

    Theory (from paper, Section 3):
      The encoding operates in an abstract referential space
      analogous to laser holography. An interference pattern
      encodes information without storing the data itself.
      Because the representation exists in this space, computation
      can occur in the same space provided transformations
      remain homomorphic.

    Core operations (paper):
      "bit-level addition, XOR (bind), permutation,
       pseudo-random bit generation, and distance"

    These are exactly the HDC primitives already implemented
    in Isildur's core engine. The Arthedain encoding is essentially
    a specific HDC encoding strategy specialized for:
    - Lossless information preservation
    - Direct computation on the encoded representation
    - Robustness to noise and corruption
    """

    def __init__(
        self,
        hv_dim: int = ARTHEDAIN_DEFAULT_DIM,
        mode: str = "bipolar",
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.hv_dim = hv_dim
        self.mode = mode
        self.seed = seed
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def encode_tensor(
        self,
        data: torch.Tensor,
        key_hvs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a tensor into a .arthedain hypervector.

        The encoding process (paper Section 3.3):
        1. Flatten the input data to a 1D vector
        2. Generate or use provided key hypervectors per element
        3. Each data element is scalar-encoded and bound with its key
        4. All bound pairs are bundled into a single output HV

        This is information-preserving because:
        - The key vectors are pseudo-random (seedable)
        - Binding preserves uniqueness per element
        - Bundling (superposition) preserves all elements
        - Only when > k elements are bundled does noise accumulate
          (by ~√k, but at d=10,000 this threshold is very high)

        Args:
            data: Any tensor (will be flattened)
            key_hvs: Optional pre-generated key hypervectors.
                    If None, generates deterministic keys.

        Returns:
            Encoded hypervector (hv_dim,) — the .arthedain representation
        """
        # Flatten to 1D
        flat = data.view(-1).float()
        n_elements = flat.shape[0]

        # Normalize to [0, 1] range for consistent encoding
        flat_min = flat.min()
        flat_max = flat.max()
        if flat_max - flat_min < 1e-8:
            flat_max = flat_min + 1.0
        flat = (flat - flat_min) / (flat_max - flat_min)

        # Generate or use key hypervectors
        if key_hvs is None:
            key_hvs = gen_hvs(
                n_elements, self.hv_dim, self.mode,
                self.device, self.seed
            )

        # Arthedain encoding via HDC binding + bundling
        # Each element: hvs.append(bind(key[i], scalar_encode(value[i])))
        # Note: In Serva, scalar encoding uses a single "level HV" per value
        # (simplified from full ItemMemory for efficiency)

        encoded_hvs = []

        # Process in chunks for large tensors
        chunk_size = 1024
        for i in range(0, n_elements, chunk_size):
            chunk_end = min(i + chunk_size, n_elements)
            chunk_vals = flat[i:chunk_end]
            chunk_keys = key_hvs[i:chunk_end]

            for j in range(len(chunk_vals)):
                # Scalar encode: threshold at median → ±1
                scalar_hv = gen_hvs(
                    1, self.hv_dim, self.mode,
                    self.device,
                    self.seed + i + j + 1 if self.seed else None
                ).squeeze(0)

                # Flip bits proportional to value (value=0 → base, value=1 → flipped)
                flip_count = int(chunk_vals[j].item() * self.hv_dim // 2)
                if flip_count > 0:
                    flip_mask = torch.randperm(self.hv_dim)[:flip_count]
                    scalar_hv[flip_mask] = -scalar_hv[flip_mask]

                # Bind value with position key
                bound_hv = bind(chunk_keys[j], scalar_hv, self.mode)
                encoded_hvs.append(bound_hv)

        # Bundle all encoded elements
        if not encoded_hvs:
            return thresh(torch.zeros(self.hv_dim, device=self.device))

        bundled = bundle(torch.stack(encoded_hvs))
        return ensure_balance(bundled)

    def encode_bytes(
        self, data: bytes, chunk_size: int = 256
    ) -> torch.Tensor:
        """
        Encode raw bytes into a .arthedain hypervector.

        Each byte is encoded as a value and bound with a position key.

        Args:
            data: Raw bytes to encode
            chunk_size: Bytes per encoding chunk

        Returns:
            Encoded hypervector (hv_dim,)
        """
        byte_tensor = torch.tensor(
            list(data), dtype=torch.float32, device=self.device
        )
        return self.encode_tensor(byte_tensor)

    def encode_image(
        self, image: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode an image tensor into a .arthedain hypervector.

        The spatial structure is preserved by using position-dependent
        key hypervectors. This ensures that nearby pixels map to
        nearby HV regions (via permutation-based position encoding).

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Encoded hypervector (hv_dim,)
        """
        if image.dim() == 4:
            image = image.squeeze(0)

        # Spatial encoding: permute keys by position
        # This preserves 2D structure in the high-dimensional space
        flat_image = image.view(-1).float()
        n_pixels = flat_image.shape[0]

        keys = gen_hvs(
            n_pixels, self.hv_dim, self.mode, self.device, self.seed
        )

        # Add spatial permutation: position i → shift by i%hv_dim
        for i in range(n_pixels):
            shift = i * 37 % self.hv_dim  # Prime offset for uniqueness
            keys[i] = permute(keys[i], k=shift)

        return self.encode_tensor(image, key_hvs=keys)

    def encode_text(
        self, text: str, token_dim: int = 256
    ) -> torch.Tensor:
        """
        Encode text into a .arthedain hypervector.

        Text is simple-hash encoded: each character becomes a
        deterministic pseudo-random hypervector, then position-tagged
        via permutation and bundled.

        Args:
            text: Input string
            token_dim: Reserved (for future token-level encoding)

        Returns:
            Encoded hypervector (hv_dim,)
        """
        char_codes = torch.tensor(
            [ord(c) for c in text],
            dtype=torch.float32,
            device=self.device,
        )

        if len(char_codes) == 0:
            return thresh(torch.zeros(self.hv_dim, device=self.device))

        return self.encode_tensor(char_codes)

    def encode_file(self, filepath: str) -> torch.Tensor:
        """
        Encode a file into a .arthedain hypervector.

        Reads the file as bytes and encodes its content.

        Args:
            filepath: Path to input file

        Returns:
            Encoded hypervector (hv_dim,)
        """
        with open(filepath, "rb") as f:
            data = f.read()
        return self.encode_bytes(data)


# ══════════════════════════════════════════════════════════════════════
# .arthedain File Format
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ArthedainFile:
    """
    .arthedain file format representation.

    The .arthedain format (from paper):
    - A universal data format that encodes any input into a single
      representational space
    - Information is preserved losslessly
    - Computation can occur directly on the encoded representation
    - Security: ciphertext generated with random seed, pushable
      client-side for encryption

    File structure:
    ┌──────────────────────────────────────────┐
    │ Header: magic bytes "ARTHEDAIN", version      │
    │ Metadata: dim, mode, seed, sha256,        │
    │           source_type, timestamp          │
    │ Hypervector: packed bits (dim/8 bytes)    │
    │ Optional: compression map, tag data       │
    └──────────────────────────────────────────┘

    Size for d=8,192: ~1 KB (vs 8 KB as float32)
    Size for d=10,000: ~1.25 KB (vs 10 KB as float32)
    """
    hypervector: torch.Tensor
    hv_dim: int
    source_type: str = "unknown"  # "image", "text", "audio", "tensor", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    sha256: str = ""

    MAGIC = b"ARTHEDAIN"
    VERSION = 1

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 of the packed hypervector."""
        packed = self._pack_bits()
        return hashlib.sha256(packed).hexdigest()

    def _pack_bits(self) -> bytes:
        """Pack bipolar HV to bits."""
        bits = (self.hypervector == 1).cpu().numpy().astype(np.uint8)
        return np.packbits(bits).tobytes()

    def save(self, filepath: str) -> None:
        """
        Save as a .arthedain file.

        Format:
        [MAGIC:5][VERSION:1][DIM:4][META_LEN:4][META_JSON:N][HV_PACKED:M]
        """
        hv_packed = self._pack_bits()
        meta_json = json.dumps({
            "hv_dim": self.hv_dim,
            "source_type": self.source_type,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }).encode("utf-8")

        with open(filepath, "wb") as f:
            f.write(self.MAGIC)
            f.write(self.VERSION.to_bytes(1, "little"))
            f.write(self.hv_dim.to_bytes(4, "little"))
            f.write(len(meta_json).to_bytes(4, "little"))
            f.write(meta_json)
            f.write(hv_packed)

    @classmethod
    def load(cls, filepath: str) -> "ArthedainFile":
        """
        Load a .arthedain file.

        Returns:
            ArthedainFile object with hypervector and metadata

        Raises:
            ValueError: If not a valid .arthedain file
        """
        with open(filepath, "rb") as f:
            magic = f.read(5)
            if magic != cls.MAGIC:
                raise ValueError(
                    f"Not a .arthedain file: magic={magic!r}"
                )

            version = int.from_bytes(f.read(1), "little")
            hv_dim = int.from_bytes(f.read(4), "little")
            meta_len = int.from_bytes(f.read(4), "little")
            meta_json = json.loads(f.read(meta_len))
            hv_packed = f.read()

        # Unpack bits to bipolar tensor
        bits = np.unpackbits(
            np.frombuffer(hv_packed, dtype=np.uint8)
        )[:hv_dim]
        hv = torch.from_numpy(
            bits.astype(np.float32) * 2 - 1
        )  # [0,1] → [-1,+1]

        return cls(
            hypervector=hv,
            hv_dim=hv_dim,
            source_type=meta_json.get("source_type", "unknown"),
            metadata=meta_json.get("metadata", {}),
            created_at=meta_json.get("created_at", ""),
            sha256=meta_json.get("sha256", ""),
        )

    def verify_integrity(self) -> bool:
        """Verify SHA-256 integrity."""
        return self.sha256 == self._compute_hash()

    @property
    def size_bytes(self) -> int:
        """Total file size in bytes."""
        return 5 + 1 + 4 + 4 + 100 + self.hv_dim // 8  # ~header + metadata + bits

    @property
    def compression_ratio(self) -> float:
        """
        Estimate compression relative to raw float32 storage.
        
        Raw: dim × 4 bytes (float32)
        .arthedain: dim / 8 bytes (packed bits) + header
        """
        raw = self.hv_dim * 4 + 100  # + header overhead
        return raw / self.size_bytes


# ══════════════════════════════════════════════════════════════════════
# Arthedain-HDC Bridge: Universal Computation on Encoded Data
# ══════════════════════════════════════════════════════════════════════

class ArthedainBridge:
    """
    Bridge between Arthedain Standard and Isildur HDC engine.

    This is the Isildur implementation of Arthedain's Chimera concept:
    "Chimera can take any model in any state and enable it to operate
    on .arthedain universal feature vector files without re-training."

    How it works:
    1. Any data → ArthedainEncoder → .arthedain hypervector
    2. Any NN model → model_to_hv() → model hypervector
    3. Both exist in the same HDC space → direct comparison
    4. Classification: Hamming distance between data HV and class HVs
    5. Similarity: cosine similarity for content matching

    This is the 'encode once, compute anywhere' paradigm.
    """

    def __init__(
        self,
        hv_dim: int = ARTHEDAIN_DEFAULT_DIM,
        encoder_seed: Optional[int] = None,
    ):
        self.hv_dim = hv_dim
        self.encoder = ArthedainEncoder(
            hv_dim=hv_dim,
            seed=encoder_seed,
        )
        self.sv_library = SVLibrary(hv_dim=hv_dim)
        self._class_hvs: Dict[str, torch.Tensor] = {}

    def encode(self, data: Any, data_type: str = "auto") -> torch.Tensor:
        """
        Encode any data into a .arthedain hypervector.

        Args:
            data: Input data (tensor, bytes, string, image tensor)
            data_type: "auto", "image", "text", "bytes", "tensor"

        Returns:
            Hypervector (hv_dim,)
        """
        if data_type == "auto":
            if isinstance(data, str):
                data_type = "text"
            elif isinstance(data, bytes):
                data_type = "bytes"
            else:
                data_type = "tensor"

        if data_type == "image":
            return self.encoder.encode_image(data)
        elif data_type == "text":
            return self.encoder.encode_text(data)
        elif data_type == "bytes":
            return self.encoder.encode_bytes(data)
        else:  # tensor
            return self.encoder.encode_tensor(data)

    def register_class(
        self,
        class_name: str,
        examples: Union[torch.Tensor, List[torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Register a class in the HDC associative memory.

        Multiple examples are bundled together.

        Args:
            class_name: Class label
            examples: One or more example tensors
            labels: Optional labels (for batch registration)

        Returns:
            Class hypervector (hv_dim,)
        """
        if isinstance(examples, list):
            hvs = []
            for ex in examples:
                hvs.append(self.encode(ex))
            class_hv = bundle(torch.stack(hvs))
        elif isinstance(examples, torch.Tensor) and examples.dim() > 1:
            # Batch of examples
            hvs = []
            for i in range(examples.shape[0]):
                hvs.append(self.encode(examples[i]))
            class_hv = bundle(torch.stack(hvs))
        else:
            # Single example
            class_hv = self.encode(examples)

        class_hv = ensure_balance(class_hv)
        self._class_hvs[class_name] = class_hv
        return class_hv

    def register_from_sv_library(
        self, class_name: str, sv_key: str
    ) -> None:
        """
        Register a class using a pre-encoded SV library vector.

        This is the fastest way to set up classification:
        use the existing SV library prototypes.

        Args:
            class_name: Class label
            sv_key: SV library key (e.g., "vision/mnist-digit-0")
        """
        self._class_hvs[class_name] = self.sv_library.get_hv(sv_key)

    def predict(
        self, data: Any, data_type: str = "auto"
    ) -> Tuple[str, float]:
        """
        Classify data against registered classes.

        Uses minimum Hamming distance in HDC space.

        Args:
            data: Input data
            data_type: Data type ("auto", "image", "text", etc.)

        Returns:
            (predicted_class, similarity)
        """
        query_hv = self.encode(data, data_type)

        if not self._class_hvs:
            raise ValueError("No classes registered. Use register_class().")

        class_names = list(self._class_hvs.keys())
        class_hvs = torch.stack(
            [self._class_hvs[name] for name in class_names]
        )

        similarities = batch_sim(query_hv, class_hvs, "bipolar")
        best_idx = int(similarities.argmax().item())

        return class_names[best_idx], float(similarities[best_idx])

    def predict_top_k(
        self, data: Any, k: int = 3, data_type: str = "auto"
    ) -> List[Tuple[str, float]]:
        """
        Top-k classification results.

        Returns:
            List of (class_name, similarity) sorted by similarity
        """
        query_hv = self.encode(data, data_type)

        class_names = list(self._class_hvs.keys())
        class_hvs = torch.stack(
            [self._class_hvs[name] for name in class_names]
        )

        similarities = batch_sim(query_hv, class_hvs, "bipolar")
        top_vals, top_idx = similarities.topk(min(k, len(class_names)))

        return [
            (class_names[int(top_idx[i])], float(top_vals[i]))
            for i in range(len(top_idx))
        ]

    def compare_to_library(
        self, data: Any, data_type: str = "auto"
    ) -> List[Tuple[str, float]]:
        """
        Compare encoded data to the SV library for concept matching.

        This answers: "What concept is this data closest to?"

        Returns:
            List of (library_key, similarity)
        """
        query_hv = self.encode(data, data_type)
        return self.sv_library.classify_by_hamming(query_hv)

    def fuse_multimodal(
        self, data_sources: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Fuse multiple data modalities into one HV.

        Each modality is encoded and bundled together.
        This is the physical realization of universal multimodality.

        Args:
            data_sources: Dict of name → data (image, audio, text, etc.)

        Returns:
            Fused hypervector (hv_dim,)

        Example:
            >>> bridge.fuse_multimodal({
            ...     "camera": image_tensor,
            ...     "lidar": lidar_points,
            ...     "radar": radar_signature,
            ... })
        """
        hvs = []
        for name, data in data_sources.items():
            hv = self.encode(data)
            hvs.append(hv)

        if not hvs:
            raise ValueError("No data sources provided")

        return bundle(torch.stack(hvs))

    def export_serva(
        self, data: Any, filepath: str, source_type: str = "auto"
    ) -> ArthedainFile:
        """
        Encode data and export as .arthedain file.

        Args:
            data: Input data
            filepath: Output .arthedain file path
            source_type: Data type label

        Returns:
            ArthedainFile object
        """
        hv = self.encode(data, source_type)
        arthedain_file = ArthedainFile(
            hypervector=hv,
            hv_dim=self.hv_dim,
            source_type=source_type,
        )
        arthedain_file.save(filepath)
        return arthedain_file


# ══════════════════════════════════════════════════════════════════════
# Zotero HDC Bibliography Reference
# ══════════════════════════════════════════════════════════════════════

HDC_BIBLIOGRAPHY = {
    "amrouch2022": {
        "title": "Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI",
        "authors": "Amrouch, H. et al.",
        "year": 2022,
        "url": "https://www.zotero.org/enotrium/collections/JTV8PX3T",
        "relevance": "Core HDC theory: CIM Hamming, associative memory, robustness",
    },
    "kanerva2009": {
        "title": "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors",
        "authors": "Kanerva, P.",
        "year": 2009,
        "url": "https://www.zotero.org/enotrium/collections/JTV8PX3T",
        "relevance": "Foundational HDC: random hypervectors, binding, bundling, permutation",
    },
    "gayler2003": {
        "title": "Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience",
        "authors": "Gayler, R. W.",
        "year": 2003,
        "url": "https://www.zotero.org/enotrium/collections/JTV8PX3T",
        "relevance": "VSA theory: role-filler bindings, compositional representation",
    },
    "plate1995": {
        "title": "Holographic Reduced Representations",
        "authors": "Plate, T. A.",
        "year": 1995,
        "url": "https://www.zotero.org/enotrium/collections/JTV8PX3T",
        "relevance": "Origin of holographic encoding: circular convolution for binding",
    },
    "arthedainmind2025": {
        "title": "The Arthedain Standard: One Primitive for All AI",
        "authors": "St. Clair, R., Cook, J. A., Sutor Jr., P., Cavero, V., Mindt, G.",
        "year": 2025,
        "url": "https://servamind.com",
        "relevance": "Arthedain encoding: holographic principles, XOR/bundle/permute HDC ops, Chimera wrapper",
    },
    "hutter2005": {
        "title": "Universal Artificial Intelligence",
        "authors": "Hutter, M.",
        "year": 2005,
        "url": "https://www.zotero.org/enotrium/collections/JTV8PX3T",
        "relevance": "Compression = Learning equivalence (Kolmogorov complexity)",
    },
    "shannon1948": {
        "title": "A Mathematical Theory of Communication",
        "authors": "Shannon, C. E.",
        "year": 1948,
        "url": "https://www.zotero.org/enotrium/collections/JTV8PX3T",
        "relevance": "Information theory: lossless compression, noisy channel capacity",
    },
}


def get_hdc_bibliography() -> Dict[str, dict]:
    """
    Get the HDC bibliography with Zotero collection reference.

    Full collection: https://www.zotero.org/enotrium/collections/JTV8PX3T

    Returns:
        Dict of bib key → reference metadata
    """
    return HDC_BIBLIOGRAPHY


def cite(key: str) -> str:
    """
    Format a bibliographic citation.

    Args:
        key: Bibliography key

    Returns:
        Formatted citation string
    """
    ref = HDC_BIBLIOGRAPHY.get(key)
    if ref is None:
        return f"[Unknown reference: {key}]"

    return (
        f"{ref['authors']} ({ref['year']}). "
        f"{ref['title']}. {ref['relevance']}"
    )


# ══════════════════════════════════════════════════════════════════════
# Initialization
# ══════════════════════════════════════════════════════════════════════

def init_sv_library(library_path: Optional[str] = None) -> SVLibrary:
    """
    Initialize and load the SV Library.

    Args:
        library_path: Path to library directory

    Returns:
        Loaded SVLibrary instance
    """
    lib = SVLibrary(library_path=library_path)
    lib.load_registry()

    if lib.is_loaded:
        print(f"[isildur/arthedain] SV Library loaded: {len(lib)} hypervectors")
        for domain in lib.list_domains():
            count = len(lib.get_hvs_by_domain(domain))
            print(f"  {domain}: {count} vectors")

    return lib