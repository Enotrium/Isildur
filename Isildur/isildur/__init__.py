"""
Isildur — Turn Any Neural Network into VSA/HDC.

Core components:
- Hypervector: HDC operations (gen_hvs, bundle, bind, permute, sim)
- model_to_hv: Convert any PyTorch model to a balanced binary hypervector
- CIMAssociativeMemory: Computing-in-Memory Hamming distance inference
- HDGlue: Hyperdimensional consensus model fusion
- FPGABackend: HLS and Verilog export for neuromorphic FPGAs
"""

from isildur.core import (
    Hypervector,
    gen_hvs,
    bundle,
    bind,
    permute,
    sim,
    thresh,
    ItemMemory,
    AssocMemory,
    SpikeHDC,
    HDCEncoder,
)

from isildur.nn_to_hdc import (
    model_to_hv,
    LayerBinarizer,
    HVComposer,
    hv_similarity,
    hv_hamming,
)

from isildur.cim import (
    CIMAssociativeMemory,
    CIMHamming,
    CIMConfig,
)

from isildur.fusion import (
    HDGlue,
    HDConsensus,
)

from isildur.fpga_backend import (
    FPGABackend,
    export_fpga_hls,
    export_verilog,
    HVOpCore,
    CIMHammingCore,
    SysBundCore,
)

from isildur.arthedain import (
    ArthedainEncoder,
    ArthedainFile,
    ArthedainBridge,
    SVLibrary,
    init_sv_library,
    get_hdc_bibliography,
    cite,
    HDC_BIBLIOGRAPHY,
    ARTHEDAIN_DEFAULT_DIM,
)

__version__ = "1.0.0"
__all__ = [
    "Hypervector",
    "gen_hvs",
    "bundle",
    "bind",
    "permute",
    "sim",
    "thresh",
    "ItemMemory",
    "AssocMemory",
    "SpikeHDC",
    "HDCEncoder",
    "model_to_hv",
    "LayerBinarizer",
    "HVComposer",
    "hv_similarity",
    "hv_hamming",
    "CIMAssociativeMemory",
    "CIMHamming",
    "CIMConfig",
    "HDGlue",
    "HDConsensus",
    "FPGABackend",
    "export_fpga_hls",
    "export_verilog",
    "HVOpCore",
    "CIMHammingCore",
    "SysBundCore",
    "ArthedainEncoder",
    "ArthedainFile",
    "ArthedainBridge",
    "SVLibrary",
    "init_sv_library",
    "get_hdc_bibliography",
    "cite",
    "HDC_BIBLIOGRAPHY",
    "ARTHEDAIN_DEFAULT_DIM",
]
