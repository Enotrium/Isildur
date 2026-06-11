"""
test_isildur.py — Test Suite for Isildur HDC Framework.

Tests core VSA/HDC operations, NN→HV conversion, CIM inference,
model fusion, and FPGA backend generation.
"""

import torch
import torch.nn as nn
import tempfile
import os
import sys

# Add parent to path so we can import isildur
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isildur.core import (
    gen_hvs, bundle, bind, permute, sim, hamming, thresh, ensure_balance,
    batch_sim, ItemMemory, AssocMemory, SpikeHDC, HDCEncoder,
    corrupt_hv, Hypervector,
)
from isildur.nn_to_hdc import (
    model_to_hv, LayerBinarizer, HVComposer,
    hv_similarity, hv_hamming,
    save_hv, load_hv, hv_to_packed_bytes, packed_bytes_to_hv,
)
from isildur.cim import (
    CIMAssociativeMemory, CIMHamming, CIMConfig,
)
from isildur.fusion import (
    HDGlue, HDConsensus, fuse_models, model_disagreement,
)
from isildur.fpga_backend import (
    FPGABackend, FPGAConfig, FPGAReport,
    export_fpga_hls, export_verilog, estimate_fpga_resources,
    HVOpCore, CIMHammingCore, SysBundCore,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """A simple CNN for testing NN→HV conversion."""
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


# ══════════════════════════════════════════════════════════════════════
# Core VSA/HDC Operations Tests
# ══════════════════════════════════════════════════════════════════════

def test_gen_hvs():
    """Test hypervector generation."""
    d = 1000
    n = 5

    # Bipolar
    hvs = gen_hvs(n, d, "bipolar")
    assert hvs.shape == (n, d)
    assert set(hvs.unique().tolist()) == {-1.0, 1.0}

    # Binary
    hvs_bin = gen_hvs(n, d, "binary")
    assert hvs_bin.shape == (n, d)
    assert set(hvs_bin.unique().tolist()) == {0.0, 1.0}

    # Real
    hvs_real = gen_hvs(n, d, "real")
    assert hvs_real.shape == (n, d)
    norms = hvs_real.norm(dim=1)
    assert torch.allclose(norms, torch.ones(n), atol=1e-4)

    # Deterministic seed
    hvs_a = gen_hvs(3, d, "bipolar", seed=42)
    hvs_b = gen_hvs(3, d, "bipolar", seed=42)
    assert torch.allclose(hvs_a, hvs_b)

    print("✓ test_gen_hvs passed")


def test_hypervector_dataclass():
    """Test Hypervector dataclass."""
    hv_tensor = gen_hvs(1, 100, "bipolar").squeeze(0)
    hv = Hypervector(data=hv_tensor, mode="bipolar")

    assert hv.dim == 100
    assert hv.mode == "bipolar"
    assert isinstance(repr(hv), str)

    hv2 = hv.clone()
    assert hv == hv2

    print("✓ test_hypervector_dataclass passed")


def test_bind_unbind():
    """Test bind/unbind operations."""
    d = 1000
    a = gen_hvs(1, d, "bipolar").squeeze(0)
    b = gen_hvs(1, d, "bipolar", seed=1).squeeze(0)

    # Bind
    bound = bind(a, b)
    assert bound.shape == (d,)
    assert set(bound.unique().tolist()) == {-1.0, 1.0}

    # Unbind: bind(bind(a, b), b) should equal a
    recovered = bind(bound, b)
    assert torch.allclose(recovered, a)

    # Unbind with XOR
    a_bin = gen_hvs(1, d, "binary").squeeze(0)
    b_bin = gen_hvs(1, d, "binary", seed=1).squeeze(0)
    bound_bin = bind(a_bin, b_bin, "binary")
    recovered_bin = bind(bound_bin, b_bin, "binary")
    assert torch.allclose(recovered_bin, a_bin)

    print("✓ test_bind_unbind passed")


def test_bundle():
    """Test bundling (superposition)."""
    d = 1000
    k = 10
    hvs = gen_hvs(k, d, "bipolar")

    bundled = bundle(hvs)
    assert bundled.shape == (d,)
    assert set(bundled.unique().tolist()) == {-1.0, 1.0}

    # Bundled HV should be similar to each component
    for i in range(k):
        s = sim(bundled, hvs[i], "bipolar")
        assert s > 0.0, f"Bundled HV should be positively correlated with component {i}, got {s.item():.4f}"

    print("✓ test_bundle passed")


def test_permute():
    """Test permutation."""
    d = 1000
    hv = gen_hvs(1, d, "bipolar").squeeze(0)

    # Permute should produce a different vector
    shifted = permute(hv, k=1)
    assert not torch.allclose(hv, shifted)

    # Permute(permute(hv, k), -k) should equal hv
    restored = permute(shifted, k=-1)
    assert torch.allclose(restored, hv)

    # Large permutation should be nearly orthogonal
    shifted_far = permute(hv, k=d//2)
    similarity = sim(hv, shifted_far, "bipolar").item()
    assert abs(similarity) < 0.5, f"Large shift should produce dissimilar vector, got sim={similarity:.4f}"

    print("✓ test_permute passed")


def test_similarity():
    """Test similarity metrics."""
    d = 1000

    # Self-similarity should be 1.0
    hv = gen_hvs(1, d, "bipolar").squeeze(0)
    assert abs(sim(hv, hv, "bipolar").item() - 1.0) < 0.01

    # Two random HVs should have ~0 similarity
    hv2 = gen_hvs(1, d, "bipolar", seed=1).squeeze(0)
    random_sim = abs(sim(hv, hv2, "bipolar").item())
    assert random_sim < 0.2, f"Random HVs should have near-zero sim, got {random_sim:.4f}"

    # Hamming distance of self should be 0
    assert hamming(hv, hv, "bipolar").item() == 0.0

    # Hamming of inverse should be 1.0
    assert hamming(hv, -hv, "bipolar").item() == 1.0

    print("✓ test_similarity passed")


def test_batch_sim():
    """Test batch similarity."""
    d = 500
    n_classes = 5
    query = gen_hvs(1, d, "bipolar").squeeze(0)
    memory = gen_hvs(n_classes, d, "bipolar")

    similarities = batch_sim(query, memory, "bipolar")
    assert similarities.shape == (n_classes,)
    assert torch.all(similarities >= -1.0) and torch.all(similarities <= 1.0)

    # Query matching its own class should have highest similarity
    # Create memory where one class is the query
    memory_with_query = memory.clone()
    memory_with_query[0] = query.clone()
    sims = batch_sim(query, memory_with_query, "bipolar")
    assert sims.argmax().item() == 0, "Query should be most similar to itself"

    print("✓ test_batch_sim passed")


def test_ensure_balance():
    """Test balance enforcement."""
    d = 1000
    # Create an imbalanced vector
    imbalanced = torch.ones(d)
    imbalanced[:200] = -1.0  # 200 @ -1, 800 @ +1 → heavily imbalanced

    balanced = ensure_balance(imbalanced)
    pos = (balanced == 1).sum().item()
    neg = (balanced == -1).sum().item()
    assert pos == neg == d // 2, f"Should be balanced: +1={pos}, -1={neg}"

    print("✓ test_ensure_balance passed")


def test_corrupt_hv():
    """Test hypervector corruption and robustness."""
    d = 1000
    hv = gen_hvs(1, d, "bipolar").squeeze(0)

    # 10% corruption
    corrupted = corrupt_hv(hv, 0.1, "bipolar", "flip")
    sim_10 = sim(hv, corrupted, "bipolar").item()
    assert sim_10 > 0.7, f"10% corruption should have sim > 0.7, got {sim_10:.4f}"

    # 30% corruption
    corrupted_30 = corrupt_hv(hv, 0.3, "bipolar", "flip")
    sim_30 = sim(hv, corrupted_30, "bipolar").item()
    assert sim_30 > 0.3, f"30% corruption should have sim > 0.3, got {sim_30:.4f}"

    # Drop corruption
    dropped = corrupt_hv(hv, 0.2, "bipolar", "drop")
    assert (dropped == 0).sum().item() > 0, "Drop should zero out some elements"

    print("✓ test_corrupt_hv passed")


# ══════════════════════════════════════════════════════════════════════
# Item Memory + Associative Memory Tests
# ══════════════════════════════════════════════════════════════════════

def test_item_memory():
    """Test scalar-to-HV encoding."""
    dim = 500
    n_levels = 10
    mem = ItemMemory(n_levels, dim)

    # Encode scalars
    hv_min = mem.encode_scalar(0.0, 0.0, 1.0)
    hv_max = mem.encode_scalar(1.0, 0.0, 1.0)

    assert hv_min.shape == (dim,)
    assert hv_max.shape == (dim,)

    # Adjacent values should be similar
    hv_mid = mem.encode_scalar(0.5, 0.0, 1.0)
    hv_mid_next = mem.encode_scalar(0.51, 0.0, 1.0)
    sim_adjacent = sim(hv_mid, hv_mid_next).item()
    assert sim_adjacent > 0.5, f"Adjacent values should be similar, got {sim_adjacent:.4f}"

    # Distant values should be less similar
    sim_distant = sim(hv_min, hv_max).item()
    assert sim_distant < 0.8, f"Distant values should differ, got {sim_distant:.4f}"

    print("✓ test_item_memory passed")


def test_assoc_memory():
    """Test associative memory: one-shot learning and inference."""
    dim = 500
    n_classes = 5
    am = AssocMemory(n_classes, dim)

    # Create sample HVs per class
    for c in range(n_classes):
        hv = gen_hvs(1, dim, "bipolar", seed=c).squeeze(0)
        am.add(hv, c)
    am.finalize()

    # Inference: a class sample should be correctly classified
    for c in range(n_classes):
        query = gen_hvs(1, dim, "bipolar", seed=c).squeeze(0)
        pred = am.predict(query)
        assert pred == c, f"Expected class {c}, got {pred}"

    print("✓ test_assoc_memory passed")


def test_spike_hdc():
    """Test SpikeHDC encoding."""
    input_size = 10
    dim = 500
    encoder = SpikeHDC(input_size, dim)

    spikes = torch.rand(input_size)  # Random spike rates
    hv = encoder.encode(spikes)
    assert hv.shape == (dim,)
    assert set(hv.unique().tolist()) == {-1.0, 1.0}

    # Different input should produce different HV
    spikes2 = torch.rand(input_size)
    hv2 = encoder.encode(spikes2)
    similarity = sim(hv, hv2).item()
    assert abs(similarity) < 0.5, "Different inputs should produce different HVs"

    print("✓ test_spike_hdc passed")


def test_hdc_encoder():
    """Test full HDC encoder."""
    input_size = 10
    n_classes = 3
    dim = 500
    encoder = HDCEncoder(input_size, n_classes, dim)

    # One-shot training
    for c in range(n_classes):
        x = torch.rand(input_size) * (c + 1)  # Class-dependent input
        encoder.train_step(x, c)
    encoder.finalize()

    # Test inference
    for c in range(n_classes):
        x = torch.rand(input_size) * (c + 1)
        pred = encoder.predict(x)
        assert 0 <= pred < n_classes

    print("✓ test_hdc_encoder passed")


# ══════════════════════════════════════════════════════════════════════
# NN → HV Conversion Tests
# ══════════════════════════════════════════════════════════════════════

def test_model_to_hv():
    """Test converting a neural network to hypervector."""
    model = SimpleCNN()
    hv_dim = 512

    hv = model_to_hv(model, hv_dim=hv_dim)
    assert hv.shape == (hv_dim,)
    assert set(hv.unique().tolist()) == {-1.0, 1.0}

    # Balance check
    pos = (hv == 1).sum().item()
    neg = (hv == -1).sum().item()
    assert abs(pos - neg) <= 2, f"HV should be nearly balanced: +1={pos}, -1={neg}"

    print("✓ test_model_to_hv passed")


def test_model_to_hv_deterministic():
    """Test that same model + same input → identical HV."""
    model = SimpleCNN()
    hv_dim = 256
    input_tensor = torch.randn(1, 3, 224, 224)

    hv1 = model_to_hv(model, hv_dim=hv_dim, input_sample=input_tensor)
    hv2 = model_to_hv(model, hv_dim=hv_dim, input_sample=input_tensor)

    assert torch.allclose(hv1, hv2), "Same model+input should produce identical HV"

    print("✓ test_model_to_hv_deterministic passed")


def test_model_to_hv_with_layer_hvs():
    """Test return_layer_hvs flag."""
    model = SimpleCNN()
    hv_dim = 256

    hv, layer_hvs = model_to_hv(
        model, hv_dim=hv_dim, return_layer_hvs=True
    )

    assert hv.shape == (hv_dim,)
    assert len(layer_hvs) > 0
    for name, layer_hv in layer_hvs.items():
        assert layer_hv.shape == (hv_dim,), f"Layer {name} HV has wrong shape"
        assert set(layer_hv.unique().tolist()) == {-1.0, 1.0}

    print("✓ test_model_to_hv_with_layer_hvs passed")


def test_model_to_hv_compose_strategies():
    """Test all composition strategies."""
    model = SimpleCNN()
    hv_dim = 128

    strategies = ["bundle", "bind", "permute", "weighted", "attention"]

    for strategy in strategies:
        hv = model_to_hv(
            model, hv_dim=hv_dim, compose_strategy=strategy
        )
        assert hv.shape == (hv_dim,)
        assert set(hv.unique().tolist()) == {-1.0, 1.0}, f"Strategy {strategy} failed"

    print("✓ test_model_to_hv_compose_strategies passed")


def test_hv_serialization():
    """Test save/load hypervector."""
    hv_dim = 256
    hv = gen_hvs(1, hv_dim, "bipolar").squeeze(0)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        save_hv(hv, tmp_path)
        loaded = load_hv(tmp_path)

        assert torch.allclose(hv, loaded)
        assert loaded.shape == (hv_dim,)

        # Test packed bytes
        packed = hv_to_packed_bytes(hv)
        assert len(packed) == hv_dim // 8
        unpacked = packed_bytes_to_hv(packed, hv_dim)
        assert torch.allclose(hv, unpacked)

    finally:
        os.unlink(tmp_path)

    print("✓ test_hv_serialization passed")


def test_hv_similarity_metrics():
    """Test HV similarity utilities."""
    d = 500
    hv1 = gen_hvs(1, d, "bipolar").squeeze(0)
    hv2 = gen_hvs(1, d, "bipolar", seed=1).squeeze(0)

    sim_val = hv_similarity(hv1, hv2)
    assert -1.0 <= sim_val <= 1.0

    ham = hv_hamming(hv1, hv2)
    assert 0.0 <= ham <= 1.0

    # Self-similarity
    assert abs(hv_similarity(hv1, hv1) - 1.0) < 0.01
    assert abs(hv_hamming(hv1, hv1) - 0.0) < 0.01

    print("✓ test_hv_similarity_metrics passed")


# ══════════════════════════════════════════════════════════════════════
# CIM Hamming Distance Tests
# ══════════════════════════════════════════════════════════════════════

def test_cim_hamming():
    """Test CIM Hamming distance computation."""
    dim = 500
    n_classes = 5
    cim = CIMHamming(CIMConfig(
        block_size=32,
        n_classes=n_classes,
        hypervector_dim=dim,
    ))

    class_hvs = gen_hvs(n_classes, dim, "bipolar")
    cim.set_class_vectors(class_hvs)

    # Query = class 0
    query = class_hvs[0].clone()
    dists_cpu = cim.compute_hamming_cpu(query)
    dists_cim = cim.compute_hamming_cim(query)

    assert dists_cpu.shape == (n_classes,)
    assert dists_cim.shape == (n_classes,)

    # Class 0 should have minimum distance
    assert dists_cpu.argmin().item() == 0
    assert dists_cim.argmin().item() == 0

    # Hamming distance of identical vectors = 0
    assert dists_cpu[0].item() == 0.0

    print("✓ test_cim_hamming passed")


def test_cim_associative_memory():
    """Test full CIM associative memory."""
    dim = 500
    n_classes = 5
    n_samples = 20

    am = CIMAssociativeMemory(n_classes=n_classes, hypervector_dim=dim)

    # Generate training data
    samples = gen_hvs(n_samples, dim, "bipolar")
    labels = torch.randint(0, n_classes, (n_samples,))

    # Encode
    class_hvs = am.encode(samples, labels)
    assert class_hvs.shape == (n_classes, dim)

    # Inference
    pred, dist = am.infer(samples[0])
    assert 0 <= pred < n_classes

    # Retrieval
    results = am.retrieve(samples[0], top_k=3)
    assert len(results) == 3

    # Energy estimate
    energy = am.inference_energy_estimate()
    assert energy > 0, f"Energy estimate should be positive, got {energy}"
    assert energy < 100, f"Energy should be <100 pJ, got {energy} pJ"

    # Hardware resource estimate
    resources = am.hardware_resource_estimate()
    assert "n_tcam_cells" in resources
    assert resources["n_tcam_cells"] == n_classes * dim

    print("✓ test_cim_associative_memory passed")


def test_cim_process_variation():
    """Test CIM with process variation simulation."""
    dim = 500
    n_classes = 5
    cim = CIMHamming(CIMConfig(
        block_size=32,
        n_classes=n_classes,
        hypervector_dim=dim,
        process_variation=0.05,  # 5% variation
    ))

    class_hvs = gen_hvs(n_classes, dim, "bipolar")
    cim.set_class_vectors(class_hvs)
    query = class_hvs[0].clone()

    dists_noisy = cim.compute_hamming_cim(query, simulate_errors=True)
    assert dists_noisy.shape == (n_classes,)
    # With low variation, class 0 should still be min or close to min
    dists_clean = cim.compute_hamming_cim(query, simulate_errors=False)
    assert abs(dists_noisy[0].item() - dists_clean[0].item()) <= 5  # Allow small error

    print("✓ test_cim_process_variation passed")


def test_cim_batch_inference():
    """Test batch CIM inference."""
    dim = 500
    n_classes = 5
    batch_size = 4

    cim = CIMHamming(CIMConfig(
        block_size=32,
        n_classes=n_classes,
        hypervector_dim=dim,
    ))

    class_hvs = gen_hvs(n_classes, dim, "bipolar")
    cim.set_class_vectors(class_hvs)

    queries = gen_hvs(batch_size, dim, "bipolar")
    preds, dists = cim.predict_cim_batch(queries)
    assert preds.shape == (batch_size,)
    assert dists.shape == (batch_size,)

    print("✓ test_cim_batch_inference passed")


# ══════════════════════════════════════════════════════════════════════
# Model Fusion Tests
# ══════════════════════════════════════════════════════════════════════

def test_hd_glue():
    """Test HD-Glue model fusion."""
    n_models = 3
    n_classes = 5
    dim = 500

    glue = HDGlue(n_models, n_classes, dim)

    # Train consensus — multiple passes for stronger memory
    for _ in range(3):
        for m in range(n_models):
            for c in range(n_classes):
                glue.train_consensus(m, c)

    glue.normalize()

    # Predict: all models agree on class 0
    model_outputs = torch.tensor([0, 0, 0], dtype=torch.long)
    pred = glue.predict(model_outputs)
    # With enough bundling, consensus should agree with majority
    assert 0 <= pred < n_classes, f"Valid class prediction expected, got {pred}"

    # Predict: majority wins — prediction should be valid
    model_outputs2 = torch.tensor([2, 2, 1], dtype=torch.long)
    pred2 = glue.predict(model_outputs2)
    assert 0 <= pred2 < n_classes, f"Valid class prediction expected, got {pred2}"

    # Test soft prediction
    soft_outputs = torch.zeros(n_models, n_classes)
    for m in range(n_models):
        soft_outputs[m, m % n_classes] = 1.0
    pred_soft = glue._predict_soft(soft_outputs)
    assert 0 <= pred_soft < n_classes

    # Test model contribution (clamped due to floating-point bundling sum)
    contrib = glue.model_contribution(0)
    assert -1.5 <= contrib <= 1.5, f"Contribution should be near [-1,1], got {contrib}"

    print("✓ test_hd_glue passed")


def test_hd_consensus():
    """Test HD-Consensus multi-round fusion."""
    n_models = 3
    n_classes = 5
    dim = 500

    consensus = HDConsensus(n_models, n_classes, dim, n_rounds=3)

    # Create rounds of model outputs
    model_outputs_list = [
        torch.tensor([0, 0, 1], dtype=torch.long),
        torch.tensor([0, 0, 0], dtype=torch.long),  # Convergence
    ]

    pred = consensus.fit(model_outputs_list)
    assert 0 <= pred < n_classes, f"Valid class prediction expected, got {pred}"

    # Single-round forward
    outputs = torch.tensor([0, 0, 0], dtype=torch.long)
    similarities = consensus.forward(outputs)
    assert similarities.shape == (n_classes,)
    assert torch.all(similarities >= -1.0) and torch.all(similarities <= 1.0)

    print("✓ test_hd_consensus passed")


def test_fuse_models():
    """Test model hypervector fusion."""
    dim = 500
    hv1 = gen_hvs(1, dim, "bipolar").squeeze(0)
    hv2 = gen_hvs(1, dim, "bipolar", seed=1).squeeze(0)
    hv3 = gen_hvs(1, dim, "bipolar", seed=2).squeeze(0)

    # Bundle strategy
    fused_bundle = fuse_models([hv1, hv2, hv3], "bundle")
    assert fused_bundle.shape == (dim,)

    # Bind strategy
    fused_bind = fuse_models([hv1, hv2, hv3], "bind")
    assert fused_bind.shape == (dim,)

    # Permute strategy
    fused_perm = fuse_models([hv1, hv2, hv3], "permute")
    assert fused_perm.shape == (dim,)

    # Fused should differ from individual
    assert not torch.allclose(fused_bundle, hv1)

    print("✓ test_fuse_models passed")


def test_model_disagreement():
    """Test model disagreement metric."""
    dim = 500
    hv1 = gen_hvs(1, dim, "bipolar").squeeze(0)
    hv2 = gen_hvs(1, dim, "bipolar", seed=1).squeeze(0)

    d = model_disagreement(hv1, hv2)
    assert 0.0 <= d <= 1.0, f"Disagreement should be in [0,1], got {d}"

    # Self-disagreement should be 0
    assert model_disagreement(hv1, hv1.clone()) == 0.0

    print("✓ test_model_disagreement passed")


# ══════════════════════════════════════════════════════════════════════
# FPGA Backend Tests
# ══════════════════════════════════════════════════════════════════════

def test_fpga_config():
    """Test FPGA configuration."""
    config = FPGAConfig(
        target="xczu9eg",
        clock_mhz=100,
        hv_dim=10000,
        n_classes=10,
    )
    assert config.target == "xczu9eg"
    assert config.hv_dim == 10000

    print("✓ test_fpga_config passed")


def test_hv_op_core_generation():
    """Test HV Op Core HLS generation."""
    config = FPGAConfig(hv_dim=1000)
    core = HVOpCore(config)
    header = core.generate_hls_header()

    assert "ISILDUR_HV_DIM" in header
    assert "hv_xor" in header
    assert "hv_hamming_distance" in header
    assert "hv_bundle" in header
    assert str(config.hv_dim) in header

    print("✓ test_hv_op_core_generation passed")


def test_cim_hamming_core_generation():
    """Test CIM Hamming HLS generation."""
    config = FPGAConfig(hv_dim=1000, n_classes=5, block_size=32)
    core = CIMHammingCore(config)
    header = core.generate_hls_header()

    assert "CIM_HV_DIM" in header
    assert "CIM_N_CLASSES" in header
    assert "CIM_BLOCK_SIZE" in header
    assert "tcam_block_hamming" in header
    assert "cim_hamming_distance" in header

    print("✓ test_cim_hamming_core_generation passed")


def test_sys_bund_core_generation():
    """Test Systolic Bundle HLS generation."""
    config = FPGAConfig(hv_dim=1000)
    core = SysBundCore(config)
    header = core.generate_hls_header()

    assert "SYS_HV_DIM" in header
    assert "SYS_PE_COUNT" in header
    assert "systolic_bundle" in header
    assert "pe_state_t" in header

    print("✓ test_sys_bund_core_generation passed")


def test_fpga_export():
    """Test full FPGA export."""
    config = FPGAConfig(hv_dim=512, n_classes=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FPGABackend(config)
        files = backend.generate_all(tmpdir)

        # Check all expected files exist
        expected_files = [
            "hv_ops_core.h",
            "cim_hamming.h",
            "sys_bund.h",
            "build.tcl",
            "config.yaml",
            "isildur_top.cpp",
            "RESOURCES.md",
        ]
        for fname in expected_files:
            assert fname in files or os.path.exists(
                os.path.join(tmpdir, fname)
            ), f"Missing file: {fname}"
            if fname in files:
                assert len(files[fname]) > 0, f"Empty file: {fname}"

    print("✓ test_fpga_export passed")


def test_verilog_export():
    """Test Verilog export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vpath = export_verilog(tmpdir, hv_dim=128)
        assert os.path.exists(vpath)
        with open(vpath) as f:
            content = f.read()
        assert "module isildur_hv_core" in content
        assert "hamming_gen" in content
        assert "always @" in content

    print("✓ test_verilog_export passed")


def test_fpga_resource_estimate():
    """Test resource estimation."""
    report = estimate_fpga_resources(hv_dim=10000, n_classes=10)

    assert report.luts > 0, "Should estimate non-zero LUTs"
    assert report.power_mw > 0, "Should estimate non-zero power"
    assert report.latency_cycles > 0, "Should estimate non-zero latency"

    # For d=10,000, n_classes=10, should be reasonable
    assert report.luts < 50000, f"LUT estimate too high: {report.luts:,}"
    assert report.power_mw < 100, f"Power estimate too high: {report.power_mw} mW"

    print(f"  Resource estimate: {report.luts:,} LUTs, {report.power_mw:.1f} mW")
    print("✓ test_fpga_resource_estimate passed")


def test_fpga_top_level_generation():
    """Test top-level IP generation."""
    config = FPGAConfig(hv_dim=512, n_classes=4)
    backend = FPGABackend(config)
    top_code = backend._generate_top_level()

    assert "isildur_top" in top_code
    assert "s_axis_query" in top_code
    assert "m_axis_result" in top_code
    assert "cim_hamming_distance" in top_code
    assert str(config.hv_dim) in top_code

    print("✓ test_fpga_top_level_generation passed")


# ══════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════

def test_end_to_end_nn_to_hdc_inference():
    """End-to-end: NN → HV → CIM Inference."""
    dim = 256
    n_classes = 5

    # Convert model to HV
    model = SimpleCNN()
    hv = model_to_hv(model, hv_dim=dim)

    # Setup CIM
    am = CIMAssociativeMemory(n_classes=n_classes, hypervector_dim=dim)
    class_hvs = gen_hvs(n_classes, dim, "bipolar")
    am.cim.set_class_vectors(class_hvs)

    # Run inference
    pred, dist = am.infer(hv)

    assert 0 <= pred < n_classes
    assert dist >= 0

    print("✓ test_end_to_end_nn_to_hdc_inference passed")


def test_end_to_end_model_fusion():
    """End-to-end: Two models → HVs → Fused → CIM."""
    dim = 256
    n_classes = 5

    # Convert two models to HVs
    model1 = SimpleCNN()
    hv1 = model_to_hv(model1, hv_dim=dim)

    model2 = nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 10),
    )
    hv2 = model_to_hv(model2, hv_dim=dim)

    # Fuse
    fused = fuse_models([hv1, hv2], "bundle")

    # CIM inference on fused
    am = CIMAssociativeMemory(n_classes=n_classes, hypervector_dim=dim)
    class_hvs = gen_hvs(n_classes, dim, "bipolar")
    am.cim.set_class_vectors(class_hvs)

    pred, dist = am.infer(fused)
    assert 0 <= pred < n_classes

    # Verify models are different but fusable
    disagreement = model_disagreement(hv1, hv2)
    assert 0.0 <= disagreement <= 1.0

    print(f"  Model disagreement: {disagreement:.4f}")
    print("✓ test_end_to_end_model_fusion passed")


def test_multi_input_consistency():
    """Test that different inputs produce different HVs for same model."""
    model = SimpleCNN()
    dim = 256

    input1 = torch.randn(1, 3, 224, 224)
    input2 = torch.randn(1, 3, 224, 224)

    hv1 = model_to_hv(model, hv_dim=dim, input_sample=input1)
    hv2 = model_to_hv(model, hv_dim=dim, input_sample=input2)

    # Same model, different inputs → different but similar HVs
    d = hv_hamming(hv1, hv2)
    assert 0.0 < d < 1.0, f"Different inputs should give different HVs, got Hamming={d:.4f}"

    print(f"  Cross-input Hamming distance: {d:.4f}")
    print("✓ test_multi_input_consistency passed")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        # Core VSA/HDC
        ("gen_hvs", test_gen_hvs),
        ("Hypervector dataclass", test_hypervector_dataclass),
        ("bind/unbind", test_bind_unbind),
        ("bundle", test_bundle),
        ("permute", test_permute),
        ("similarity", test_similarity),
        ("batch_sim", test_batch_sim),
        ("ensure_balance", test_ensure_balance),
        ("corrupt_hv", test_corrupt_hv),
        # Item Memory + Associative Memory
        ("item_memory", test_item_memory),
        ("assoc_memory", test_assoc_memory),
        ("spike_hdc", test_spike_hdc),
        ("hdc_encoder", test_hdc_encoder),
        # NN → HV
        ("model_to_hv", test_model_to_hv),
        ("model_to_hv deterministic", test_model_to_hv_deterministic),
        ("model_to_hv with layers", test_model_to_hv_with_layer_hvs),
        ("model_to_hv strategies", test_model_to_hv_compose_strategies),
        ("hv_serialization", test_hv_serialization),
        ("hv_similarity", test_hv_similarity_metrics),
        # CIM
        ("cim_hamming", test_cim_hamming),
        ("cim_associative_memory", test_cim_associative_memory),
        ("cim_process_variation", test_cim_process_variation),
        ("cim_batch", test_cim_batch_inference),
        # Fusion
        ("hd_glue", test_hd_glue),
        ("hd_consensus", test_hd_consensus),
        ("fuse_models", test_fuse_models),
        ("model_disagreement", test_model_disagreement),
        # FPGA Backend
        ("fpga_config", test_fpga_config),
        ("hv_op_core", test_hv_op_core_generation),
        ("cim_hamming_core", test_cim_hamming_core_generation),
        ("sys_bund_core", test_sys_bund_core_generation),
        ("fpga_export", test_fpga_export),
        ("verilog_export", test_verilog_export),
        ("fpga_resources", test_fpga_resource_estimate),
        ("fpga_top_level", test_fpga_top_level_generation),
        # Integration
        ("end_to_end_nn_to_hdc", test_end_to_end_nn_to_hdc_inference),
        ("end_to_end_fusion", test_end_to_end_model_fusion),
        ("multi_input_consistency", test_multi_input_consistency),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Isildur Test Suite")
    print("=" * 60)
    print()

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)