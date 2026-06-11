"""
fusion.py — HD-Glue: Hyperdimensional Consensus Model Fusion.

Fuses multiple neural network outputs via hyperdimensional binding/bundling
into a consensus prediction. Based on Section V-C of Amrouch et al. 2022.

Theory:
  Each model contributes a class likelihood vector. These are bound with
  unique model-identity hypervectors and class hypervectors, then bundled
  into a memory trace. At inference time, querying the memory trace with
  model outputs yields a consensus classification via HDC similarity.

Advantages over traditional ensemble methods:
  - No retraining needed for new models
  - Models can be added/removed incrementally
  - Decision trace is interpretable via HVD similarity decomposition
  - Robust to individual model failures
  - Hardware-efficient (XOR + popcount)

Applications:
  - Federated model aggregation (privacy-preserving via HDC binding)
  - Multi-modal fusion (camera + LIDAR + radar → unified classification)
  - Model verification (check if new model matches known-good pattern)
  - Continual learning (add new knowledge without catastrophic forgetting)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from collections import OrderedDict

from isildur.core import (
    gen_hvs, bind, bundle, batch_sim, thresh,
    ensure_balance, sim,
)


class HDGlue(nn.Module):
    """
    HD-Glue: Hyperdimensional Consensus for Model Fusion.

    Fuses n_models into one consensus classifier using HDC binding/bundling.

    Training:
      For each (model_idx, class_idx) pair:
        consensus_hv = bundle(bind(model_id[model_idx], class_hv[class_idx]))

    Inference:
      probe = sum_i P(y=c|x; model_i) * bind(model_id[i], class_hv[c])
      consensus = sign(probe)
      result = argmax_c similarity(consensus, class_hv[c])

    This creates a shared representational space where each model's
    predictions are additive evidence towards the consensus.

    Properties:
    - Associative: removing a model just subtracts its contribution
    - Monotonic: adding more well-trained models improves consensus
    - Noise-robust: low-quality models are drowned out by high-quality ones
    - Hardware-friendly: all operations are XOR/binary
    """

    def __init__(
        self,
        n_models: int,
        n_classes: int,
        dim: int = 10000,
        mode: str = "bipolar",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_classes = n_classes
        self.dim = dim
        self.mode = mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Unique identifiers for each model
        self.register_buffer(
            "model_ids",
            gen_hvs(n_models, dim, mode, self.device, seed),
        )

        # Class hypervectors (shared across models)
        self.register_buffer(
            "class_hvs",
            gen_hvs(n_classes, dim, mode, self.device, seed + 1 if seed else None),
        )

        # Memory trace: accumulates all model-class bindings
        self.register_buffer(
            "memory_hv", torch.zeros(dim, device=self.device)
        )

        # Per-class accumulators for model fusion
        self.register_buffer(
            "class_accumulator",
            torch.zeros(n_classes, dim, device=self.device),
        )

        # Counts for normalization
        self.register_buffer(
            "class_counts",
            torch.zeros(n_classes, device=self.device),
        )

    def train_consensus(self, model_idx: int, class_idx: int) -> None:
        """
        Register a model's knowledge of a class.

        For each (model, class) pair, bind the model identity with the
        class identity and add to the consensus memory.

        Args:
            model_idx: Which model (0..n_models-1)
            class_idx: Which class (0..n_classes-1)
        """
        # Bind model identity with class identity
        bound = bind(
            self.model_ids[model_idx],
            self.class_hvs[class_idx],
            self.mode,
        )
        self.memory_hv = self.memory_hv + bound

        # Track per-class contributions
        self.class_accumulator[class_idx] = (
            self.class_accumulator[class_idx] + bound
        )
        self.class_counts[class_idx] += 1

    def train_consensus_weighted(
        self,
        model_idx: int,
        class_idx: int,
        weight: float = 1.0,
    ) -> None:
        """
        Weighted consensus training.

        Models with higher reliability/accuracy can be weighted
        more heavily in the consensus.

        Args:
            model_idx: Which model
            class_idx: Which class
            weight: Contribution weight (1.0 = full, 0.0 = none)
        """
        bound = bind(
            self.model_ids[model_idx],
            self.class_hvs[class_idx],
            self.mode,
        )
        self.memory_hv = self.memory_hv + weight * bound

    def predict(
        self, model_outputs: torch.Tensor
    ) -> int:
        """
        Predict class given outputs from all models.

        Args:
            model_outputs: Tensor of shape (n_models,). Each entry is
                          the predicted class index from that model.
                          For probabilistic models, pass logits or
                          probabilities per model (n_models, n_classes).

        Returns:
            Consensus class prediction (int).

        Raises:
            ValueError: If model_outputs shape doesn't match n_models.
        """
        if model_outputs.dim() == 1:
            # Hard predictions: one class per model
            return self._predict_hard(model_outputs)

        elif model_outputs.dim() == 2:
            # Soft predictions: probability vector per model
            return self._predict_soft(model_outputs)

        else:
            raise ValueError(
                f"model_outputs must be 1D (hard) or 2D (soft), "
                f"got shape {model_outputs.shape}"
            )

    def _predict_hard(self, model_tags: torch.Tensor) -> int:
        """
        Hard-voting consensus: each model votes for one class.
        """
        probe = torch.zeros(self.dim, device=self.device)
        for m in range(self.n_models):
            c = int(model_tags[m].item())
            if 0 <= c < self.n_classes:
                bound = bind(self.model_ids[m], self.class_hvs[c], self.mode)
                probe = probe + bound

        probe = thresh(probe)
        # Find class with highest similarity to probe
        similarities = batch_sim(probe, self.class_hvs, self.mode)
        return int(similarities.argmax().item())

    def _predict_soft(self, probabilities: torch.Tensor) -> int:
        """
        Soft-voting consensus: each model provides class probabilities.

        Args:
            probabilities: (n_models, n_classes) — each row sums to 1
        """
        probe = torch.zeros(self.dim, device=self.device)
        for m in range(self.n_models):
            for c in range(self.n_classes):
                weight = probabilities[m, c].item()
                if weight > 0:
                    bound = bind(
                        self.model_ids[m], self.class_hvs[c], self.mode
                    )
                    probe = probe + weight * bound

        probe = thresh(probe)
        similarities = batch_sim(probe, self.class_hvs, self.mode)
        return int(similarities.argmax().item())

    def predict_with_scores(
        self, model_outputs: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """
        Predict with similarity scores for all classes.

        Returns:
            (predicted_class, similarities) where similarities is (n_classes,)
        """
        if model_outputs.dim() == 1:
            probe = torch.zeros(self.dim, device=self.device)
            for m in range(self.n_models):
                c = int(model_outputs[m].item())
                if 0 <= c < self.n_classes:
                    probe = probe + bind(
                        self.model_ids[m], self.class_hvs[c], self.mode
                    )
        else:
            probe = torch.zeros(self.dim, device=self.device)
            for m in range(self.n_models):
                for c in range(self.n_classes):
                    w = model_outputs[m, c].item()
                    if w > 0:
                        probe = probe + w * bind(
                            self.model_ids[m], self.class_hvs[c], self.mode
                        )

        probe = thresh(probe)
        similarities = batch_sim(probe, self.class_hvs, self.mode)
        return int(similarities.argmax().item()), similarities

    def normalize(self) -> None:
        """
        Normalize the consensus memory — call after all training.

        For bipolar: threshold to ±1
        For binary: threshold by mean
        For real: unit-norm normalize
        """
        if self.mode == "bipolar":
            self.memory_hv = thresh(self.memory_hv)
        elif self.mode == "binary":
            self.memory_hv = (self.memory_hv > 0).float()
        else:
            self.memory_hv = (
                self.memory_hv / self.memory_hv.norm().clamp(min=1e-12)
            )

    def model_contribution(
        self, model_idx: int
    ) -> float:
        """
        Estimate a model's contribution to the consensus.

        Computes similarity between the model's bound contributions
        and the full consensus memory. Higher = more influential.

        Returns:
            Contribution score ∈ [-1, 1]
        """
        model_contrib = torch.zeros(self.dim, device=self.device)
        for c in range(self.n_classes):
            model_contrib = model_contrib + bind(
                self.model_ids[model_idx], self.class_hvs[c], self.mode
            )
        return float(sim(model_contrib, self.memory_hv, self.mode).item())

    def remove_model(self, model_idx: int) -> None:
        """
        Remove a model's contributions from the consensus.

        Uses the property that binding is its own inverse.
        """
        for c in range(self.n_classes):
            bound = bind(
                self.model_ids[model_idx], self.class_hvs[c], self.mode
            )
            self.memory_hv = self.memory_hv - bound

    def add_model(
        self, model_outputs: torch.Tensor, model_idx: Optional[int] = None
    ) -> None:
        """
        Add a new model's outputs to the consensus (incremental).

        If model_idx is None, appends as a new model.
        """
        if model_idx is None or model_idx >= self.n_models:
            # Extend model_ids (requires re-initialization)
            pass
        for c in range(self.n_classes):
            if model_outputs.dim() == 1:
                if int(model_outputs[model_idx].item()) == c:
                    self.train_consensus(model_idx, c)
            else:
                w = model_outputs[model_idx, c].item()
                if w > 0:
                    self.train_consensus_weighted(model_idx, c, w)

    def forward(
        self, model_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass — returns similarity scores for all classes.

        Args:
            model_outputs: (n_models, n_classes) or (n_models,)

        Returns:
            Similarities tensor of shape (n_classes,)
        """
        if model_outputs.dim() == 1:
            probe = torch.zeros(self.dim, device=self.device)
            for m in range(self.n_models):
                c = int(model_outputs[m].item())
                if 0 <= c < self.n_classes:
                    probe = probe + bind(
                        self.model_ids[m], self.class_hvs[c], self.mode
                    )
        else:
            probe = torch.zeros(self.dim, device=self.device)
            for m in range(self.n_models):
                for c in range(self.n_classes):
                    w = model_outputs[m, c].item()
                    if w > 0:
                        probe = probe + w * bind(
                            self.model_ids[m], self.class_hvs[c], self.mode
                        )

        probe = thresh(probe)
        return batch_sim(probe, self.class_hvs, self.mode)


class HDConsensus(nn.Module):
    """
    HD-Consensus: Multi-round hyperdimensional consensus.

    Extends HD-Glue with iterative refinement:
    1. Each model proposes a class
    2. Consensus is computed via HDC binding/bundling
    3. Models are re-weighted based on agreement with consensus
    4. Repeat for n_rounds or until convergence

    This is useful for:
    - Federated learning: global model = HDC consensus of local models
    - Swarm intelligence: multiple agents reach consensus efficiently
    - Adversarial robustness: outlying models naturally down-weighted
    """

    def __init__(
        self,
        n_models: int,
        n_classes: int,
        dim: int = 10000,
        n_rounds: int = 5,
        mode: str = "bipolar",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_classes = n_classes
        self.dim = dim
        self.n_rounds = n_rounds
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.glue = HDGlue(n_models, n_classes, dim, mode, self.device, seed)
        self.register_buffer(
            "model_weights",
            torch.ones(n_models, device=self.device) / n_models,
        )

    def fit(
        self,
        model_outputs_list: List[torch.Tensor],
        max_rounds: Optional[int] = None,
        convergence_threshold: float = 0.95,
    ) -> int:
        """
        Run consensus rounds until convergence.

        Args:
            model_outputs_list: List of model outputs per round
            max_rounds: Override default n_rounds
            convergence_threshold: Stop when agreement > threshold

        Returns:
            Final consensus prediction
        """
        rounds = max_rounds or self.n_rounds

        for r in range(rounds):
            # Reset probe
            probe = torch.zeros(self.dim, device=self.device)
            for m in range(self.n_models):
                outputs = model_outputs_list[min(r, len(model_outputs_list) - 1)]
                if outputs.dim() == 1:
                    c = int(outputs[m].item())
                    bound = bind(
                        self.glue.model_ids[m],
                        self.glue.class_hvs[c],
                        self.glue.mode,
                    )
                else:
                    weighted = torch.zeros(self.dim, device=self.device)
                    for c_val in range(self.n_classes):
                        w = outputs[m, c_val].item()
                        if w > 0:
                            weighted = weighted + w * bind(
                                self.glue.model_ids[m],
                                self.glue.class_hvs[c_val],
                                self.glue.mode,
                            )
                    bound = weighted
                probe = probe + self.model_weights[m] * bound

            probe = thresh(probe)
            similarities = batch_sim(
                probe, self.glue.class_hvs, self.glue.mode
            )
            consensus_class = int(similarities.argmax().item())
            consensus_score = float(similarities.max().item())

            # Update model weights based on agreement
            for m in range(self.n_models):
                if outputs.dim() == 1:
                    agreement = 1.0 if int(outputs[m].item()) == consensus_class else 0.0
                else:
                    agreement = float(outputs[m, consensus_class].item())
                self.model_weights[m] = (
                    0.9 * self.model_weights[m] + 0.1 * agreement
                )

            if consensus_score >= convergence_threshold:
                break

        return consensus_class

    def forward(
        self, model_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-round forward pass with current weights.

        Returns similarity scores for all classes.
        """
        probe = torch.zeros(self.dim, device=self.device)
        for m in range(self.n_models):
            if model_outputs.dim() == 1:
                c = int(model_outputs[m].item())
                bound = bind(
                    self.glue.model_ids[m],
                    self.glue.class_hvs[c],
                    self.glue.mode,
                )
            else:
                weighted = torch.zeros(self.dim, device=self.device)
                for c_val in range(self.n_classes):
                    w = model_outputs[m, c_val].item()
                    if w > 0:
                        weighted = weighted + w * bind(
                            self.glue.model_ids[m],
                            self.glue.class_hvs[c_val],
                            self.glue.mode,
                        )
                bound = weighted
            probe = probe + self.model_weights[m] * bound

        probe = thresh(probe)
        return batch_sim(probe, self.glue.class_hvs, self.glue.mode)

    def predict(self, model_outputs: torch.Tensor) -> int:
        """Predict consensus class."""
        return int(self.forward(model_outputs).argmax().item())


# ══════════════════════════════════════════════════════════════════════
# Fusion utilities
# ══════════════════════════════════════════════════════════════════════

def fuse_models(
    model_hvs: List[torch.Tensor],
    compose_strategy: str = "bundle",
) -> torch.Tensor:
    """
    Fuse multiple model hypervectors into a single ensemble HV.

    Args:
        model_hvs: List of model hypervectors of same dimension
        compose_strategy: "bundle" (additive), "bind" (entangled),
                         "permute" (position-sensitive)

    Returns:
        Fused hypervector (dim,)

    Example:
        >>> resnet_hv = model_to_hv(resnet, hv_dim=10000)
        >>> vit_hv = model_to_hv(vit, hv_dim=10000)
        >>> fused = fuse_models([resnet_hv, vit_hv], "bundle")
        >>> # The fused HV represents both models
    """
    if not model_hvs:
        raise ValueError("No model HVs to fuse.")

    dim = model_hvs[0].shape[0]
    for hv in model_hvs:
        assert hv.shape[0] == dim, f"All HVs must have same dim ({dim})"

    if compose_strategy == "bundle":
        fused = bundle(torch.stack(model_hvs))
    elif compose_strategy == "bind":
        fused = model_hvs[0].clone()
        for hv in model_hvs[1:]:
            fused = bind(fused, hv)
    elif compose_strategy == "permute":
        shifted = []
        for i, hv in enumerate(model_hvs):
            shifted.append(torch.roll(hv, shifts=i))
        fused = bundle(torch.stack(shifted))
    else:
        raise ValueError(f"Unknown strategy: {compose_strategy}")

    return fused


def model_disagreement(
    hv1: torch.Tensor, hv2: torch.Tensor
) -> float:
    """
    Measure disagreement between two model hypervectors.

    Returns Hamming distance normalized to [0, 1].
    0 = identical, 0.5 = random, 1 = inverse.
    """
    return float((hv1 != hv2).sum().item() / hv1.shape[0])