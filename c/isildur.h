/**
 * isildur.h — Isildur HDC/VSA Core Library (C Reference Implementation)
 *
 * Portable C99 implementation of Hyperdimensional Computing operations.
 * No dependencies beyond libc. Compiles on any platform.
 *
 * Operations:
 *   - bind (XOR): element-wise XOR of two hypervectors
 *   - bundle (superposition): accumulate + sign-threshold
 *   - hamming_distance: popcount of XOR
 *   - associative_infer: find closest class by Hamming distance
 *   - gen_hv: generate random bipolar hypervector
 *   - item_memory: scalar → hypervector quantization
 *   - cim: computing-in-memory hamming distance (TCAM model)
 *   - fusion: HD-Glue consensus model fusion
 *   - noise: bit-flip / drop / scale corruption
 *   - encode: SpikeHDC / full HDC encoder
 *
 * Hypervectors are stored as uint64_t arrays (packed bits).
 * For d=10,000: 157 uint64_t words (1,256 bytes).
 *
 * Hardware mapping:
 *   - XOR: 1 gate per bit, single cycle
 *   - Popcount: reduction tree, log(d) cycles
 *   - Bundle: accumulate + sign-threshold, d cycles per vector
 *   - Hamming: XOR + popcount, log(d) cycles
 *
 * Based on:
 *   - Kanerva 2009: "Hyperdimensional Computing"
 *   - Amrouch et al. 2022: "Brain-Inspired HDC for Ultra-Efficient Edge AI"
 */

#ifndef ISILDUR_H
#define ISILDUR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ──────────────────────────────────────────────── */

/** Maximum supported hypervector dimension. */
#define ISILDUR_MAX_DIM 16384

/** Number of 64-bit words needed for d bits. */
#define ISILDUR_WORDS(d) (((d) + 63) / 64)

/** Default hypervector dimension (Kanerva 2009: 10,000 bits). */
#define ISILDUR_DEFAULT_DIM 10000

/** Default number of classes for associative memory. */
#define ISILDUR_MAX_CLASSES 256

/** Default number of item-memory levels. */
#define ISILDUR_DEFAULT_LEVELS 13

/* ── Hypervector Representation ─────────────────────────────────── */

/**
 * A bipolar hypervector stored as packed bits.
 *
 * Each bit position i corresponds to:
 *   word = i / 64, shift = i % 64
 *   1 bit = +1 (bipolar positive)
 *   0 bit = -1 (bipolar negative)
 *
 * For d=10,000: 157 words, 1,256 bytes.
 * Operations are all O(d/64) word-parallel.
 */
typedef struct {
    uint64_t *bits;          /** Packed bit array (caller-allocated) */
    uint32_t  dim;           /** Hypervector dimension in bits */
    uint32_t  n_words;       /** Number of uint64_t words */
} isildur_hv_t;

/* ── RNG State (exposed for multi-seed scenarios) ───────────────── */

typedef struct {
    uint64_t state;
} isildur_rng_t;

/** Initialize an RNG with a seed (0 = time-based). */
void isildur_rng_init(isildur_rng_t *rng, uint64_t seed);

/** Generate next 64-bit random value. */
uint64_t isildur_rng_next(isildur_rng_t *rng);

/* ── Lifecycle ──────────────────────────────────────────────────── */

isildur_hv_t *isildur_alloc_hv(uint32_t dim);
void isildur_free_hv(isildur_hv_t *hv);

/** Clone (deep copy) a hypervector. */
isildur_hv_t *isildur_clone_hv(const isildur_hv_t *src);

/* ── Generation ─────────────────────────────────────────────────── */

/**
 * Generate a random bipolar hypervector using an explicit RNG.
 * For backwards compatibility, passing NULL for rng uses an internal LCG.
 */
void isildur_gen_hv_rng(isildur_hv_t *hv, isildur_rng_t *rng);

/** Generate a random bipolar hypervector (legacy: uses global LCG). */
void isildur_gen_hv(isildur_hv_t *hv, uint64_t seed);

/**
 * Generate n random hypervectors into pre-allocated array.
 * Each vector gets its own deterministic seed derived from base_seed + index.
 */
void isildur_gen_hv_batch(isildur_hv_t **hvs, uint32_t n, uint64_t base_seed);

void isildur_gen_balanced_hv(isildur_hv_t *hv, uint64_t seed);
void isildur_balance(isildur_hv_t *hv);

/* ── Core HDC Operations ────────────────────────────────────────── */

void isildur_bind(isildur_hv_t *result,
                  const isildur_hv_t *a,
                  const isildur_hv_t *b);

void isildur_unbind(isildur_hv_t *result,
                    const isildur_hv_t *bound,
                    const isildur_hv_t *key);

void isildur_bundle(isildur_hv_t *result,
                    const isildur_hv_t **hvs,
                    uint32_t k);

/**
 * Accumulate one HV into a running bundle.
 * total[k] += hvs[k] bit by bit; counts is incremented.
 * Faster than full bundle() for incremental training.
 */
void isildur_bundle_accumulate(const isildur_hv_t **accums, uint32_t *counts,
                               const isildur_hv_t *hv, uint32_t n_accums,
                               uint32_t class_idx);

/**
 * Finalize accumulated bundles via sign-threshold.
 * result[i] = sign(accums[i]) with counts[i].
 */
void isildur_bundle_finalize(isildur_hv_t *result,
                             const isildur_hv_t **accums,
                             const uint32_t *counts,
                             uint32_t class_idx);

/**
 * Bundling over int32 accumulators (for batched training).
 * accums[i] is a per-bit int32 running sum. count is # of vectors bundled so far.
 * After accumulation, threshold to produce final HV.
 */
void isildur_bundle_int32_accumulate(int32_t *accums,
                                     const isildur_hv_t *hv,
                                     uint32_t dim, uint32_t *count);

void isildur_bundle_int32_finalize(isildur_hv_t *result,
                                   const int32_t *accums,
                                   uint32_t dim, uint32_t count);

/**
 * Optimised permute: cyclic right-shift at word level.
 */
void isildur_permute(isildur_hv_t *result,
                     const isildur_hv_t *hv,
                     uint32_t shift);

/* ── Similarity ─────────────────────────────────────────────────── */

uint32_t isildur_hamming(const isildur_hv_t *a,
                          const isildur_hv_t *b);

int32_t isildur_similarity(const isildur_hv_t *a,
                            const isildur_hv_t *b);

/**
 * Batch Hamming distance: query vs each of n_candidates.
 * @param distances  Output array of length n_candidates (caller-allocated).
 */
void isildur_hamming_batch(const isildur_hv_t *query,
                           const isildur_hv_t **candidates,
                           uint32_t n_candidates,
                           uint32_t *distances);

/* ── Popcount ───────────────────────────────────────────────────── */

uint32_t isildur_popcount(const isildur_hv_t *hv);

/* ── Associative Memory ─────────────────────────────────────────── */

uint32_t isildur_assoc_infer(const isildur_hv_t *query,
                              const isildur_hv_t **class_hvs,
                              uint32_t n_classes,
                              uint32_t *distances);

void isildur_assoc_topk(const isildur_hv_t *query,
                         const isildur_hv_t **class_hvs,
                         uint32_t n_classes,
                         uint32_t k,
                         uint32_t *top_indices,
                         uint32_t *top_dists);

void isildur_assoc_train(isildur_hv_t *class_hv,
                          const isildur_hv_t *sample_hv,
                          uint32_t *count);

/* ── Utility ────────────────────────────────────────────────────── */

void isildur_copy(isildur_hv_t *dst, const isildur_hv_t *src);
bool isildur_equals(const isildur_hv_t *a, const isildur_hv_t *b);
size_t isildur_serialize(const isildur_hv_t *hv,
                           uint8_t *buf, size_t buf_size);
isildur_hv_t *isildur_deserialize(const uint8_t *buf,
                                    size_t buf_size, uint32_t dim);
void isildur_print(const isildur_hv_t *hv, uint32_t max_bits);

/**
 * Set a single bit in a hypervector.
 */
void isildur_bit_set(isildur_hv_t *hv, uint32_t i, uint64_t v);

/**
 * Get a single bit from a hypervector.
 */
uint64_t isildur_bit_get(const isildur_hv_t *hv, uint32_t i);

/* ── Noise / Corruption ─────────────────────────────────────────── */

typedef enum {
    ISILDUR_NOISE_FLIP  = 0,  /** Invert random bits */
    ISILDUR_NOISE_DROP  = 1,  /** Zero out random bits */
    ISILDUR_NOISE_SCALE = 2   /** Random sign flip (same as FLIP for bipolar) */
} isildur_noise_type_t;

/**
 * Corrupt a hypervector by flipping/dropping a fraction of bits.
 *
 * @param result    Output corrupted HV (can be same as src for in-place)
 * @param src       Original hypervector
 * @param rate      Fraction of bits to corrupt [0.0, 1.0]
 * @param type      Type of noise
 * @param rng       RNG for reproducibility (NULL = time-based)
 */
void isildur_corrupt_hv(isildur_hv_t *result,
                        const isildur_hv_t *src,
                        float rate,
                        isildur_noise_type_t type,
                        isildur_rng_t *rng);

/* ── Item Memory (scalar → HV quantization) ─────────────────────── */

typedef struct {
    isildur_hv_t **level_hvs;   /** Array of n_levels hypervectors */
    uint32_t       n_levels;    /** Number of quantization levels */
    uint32_t       dim;         /** Hypervector dimension */
} isildur_itemmem_t;

/**
 * Create an item memory with n_levels quantization levels.
 * Adjacent levels are partially correlated (~flip_rate fraction of bits differ).
 *
 * @param n_levels   Number of quantization levels (e.g., 13)
 * @param dim        Hypervector dimension
 * @param seed       Random seed
 * @param flip_rate  Fraction of bits flipped between adjacent levels (0.0-0.5)
 * @return           Allocated item memory (caller must free)
 */
isildur_itemmem_t *isildur_itemmem_create(uint32_t n_levels, uint32_t dim,
                                           uint64_t seed, float flip_rate);

/** Free an item memory and all its hypervectors. */
void isildur_itemmem_free(isildur_itemmem_t *im);

/**
 * Encode a scalar value into a hypervector by quantizing to nearest level.
 *
 * @param im       Item memory
 * @param value    Scalar value
 * @param min_val  Minimum of valid range
 * @param max_val  Maximum of valid range
 * @return         Newly allocated hypervector (caller must free)
 */
isildur_hv_t *isildur_itemmem_encode_scalar(const isildur_itemmem_t *im,
                                              float value,
                                              float min_val, float max_val);

/**
 * Encode a vector of scalar values using key hypervectors.
 * For each i: bound(key_hvs[i], encode_scalar(values[i])), then bundle all.
 *
 * @param im        Item memory
 * @param values    Scalar values array
 * @param n_values  Number of values
 * @param key_hvs   Key hypervectors (one per value)
 * @param min_val   Minimum value
 * @param max_val   Maximum value
 * @return          Newly allocated bundled hypervector (caller must free)
 */
isildur_hv_t *isildur_itemmem_encode_vector(const isildur_itemmem_t *im,
                                              const float *values,
                                              uint32_t n_values,
                                              isildur_hv_t **key_hvs,
                                              float min_val, float max_val);

/**
 * Get a pointer to a specific level hypervector (no copy).
 */
const isildur_hv_t *isildur_itemmem_get_level(const isildur_itemmem_t *im,
                                                uint32_t level);

/* ── CIM Hamming Distance ───────────────────────────────────────── */

typedef struct {
    uint32_t block_size;         /** Bits per TCAM block */
    uint32_t n_classes;          /** Number of stored class vectors */
    uint32_t hv_dim;             /** Hypervector dimension */
    uint32_t n_blocks;           /** Number of TCAM blocks */
    float    process_variation;  /** Process variation [0.0, 1.0] */
} isildur_cim_config_t;

typedef struct {
    isildur_cim_config_t config;
    isildur_hv_t        **class_hvs;  /** Stored class hypervectors (binary) */
} isildur_cim_t;

/**
 * Create a CIM Hamming distance accelerator.
 *
 * @param config  CIM configuration (NULL = defaults: block_size=32)
 * @return        Allocated CIM (caller must free)
 */
isildur_cim_t *isildur_cim_create(const isildur_cim_config_t *config);

/** Free a CIM accelerator. */
void isildur_cim_free(isildur_cim_t *cim);

/**
 * Load class hypervectors into CIM memory array.
 * Class vectors are binarized (thresholded at 0) on load.
 *
 * @param cim        CIM accelerator
 * @param class_hvs  Array of n_classes bipolar hypervectors
 * @param n_classes  Number of classes
 */
void isildur_cim_load(isildur_cim_t *cim,
                      isildur_hv_t **class_hvs,
                      uint32_t n_classes);

/**
 * Compute Hamming distances via block-parallel CIM simulation.
 *
 * @param cim     CIM accelerator
 * @param query   Query hypervector
 * @param dists   Output: Hamming distances to all classes (caller-allocated, n_classes)
 */
void isildur_cim_hamming(const isildur_cim_t *cim,
                         const isildur_hv_t *query,
                         uint32_t *distances);

/**
 * CIM prediction: find class with minimum Hamming distance.
 *
 * @param cim    CIM accelerator
 * @param query  Query hypervector
 * @param dist   Output: minimum distance (nullable)
 * @return       Index of closest class
 */
uint32_t isildur_cim_predict(const isildur_cim_t *cim,
                              const isildur_hv_t *query,
                              uint32_t *dist);

/**
 * CIM batch prediction.
 *
 * @param cim      CIM accelerator
 * @param queries  Array of n_queries hypervectors
 * @param n        Number of queries
 * @param preds    Output: predicted class for each query (caller-allocated, n)
 * @param dists    Output: distances for each prediction (nullable)
 */
void isildur_cim_predict_batch(const isildur_cim_t *cim,
                               isildur_hv_t **queries,
                               uint32_t n,
                               uint32_t *predictions,
                               uint32_t *distances);

/**
 * Estimate CIM inference energy in picojoules.
 * Based on TCAM models from Amrouch et al. 2022.
 */
float isildur_cim_energy_pj(const isildur_cim_t *cim);

/* ── HD-Glue: Model Fusion ──────────────────────────────────────── */

typedef struct {
    isildur_hv_t **model_ids;     /** Model identity hypervectors [n_models] */
    isildur_hv_t **class_hvs;     /** Class hypervectors [n_classes] */
    isildur_hv_t  *memory_hv;     /** Consensus memory trace */
    isildur_hv_t **class_accum;   /** Per-class accumulators [n_classes] */
    uint32_t      *class_counts;  /** Per-class sample counts */
    uint32_t       n_models;
    uint32_t       n_classes;
    uint32_t       dim;
} isildur_fusion_t;

/**
 * Create an HD-Glue fusion instance.
 *
 * @param n_models   Number of models to fuse
 * @param n_classes  Number of classes
 * @param dim        Hypervector dimension
 * @param seed       Random seed for model IDs and class HVs
 * @return           Allocated fusion instance (caller must free)
 */
isildur_fusion_t *isildur_fusion_create(uint32_t n_models,
                                         uint32_t n_classes,
                                         uint32_t dim,
                                         uint64_t seed);

/** Free a fusion instance. */
void isildur_fusion_free(isildur_fusion_t *f);

/**
 * Register a model's knowledge of a class (train consensus).
 *
 * @param f          Fusion instance
 * @param model_idx  Which model (0..n_models-1)
 * @param class_idx  Which class (0..n_classes-1)
 * @param weight     Contribution weight (1.0 = full, 0.0 = none)
 */
void isildur_fusion_train(isildur_fusion_t *f,
                           uint32_t model_idx,
                           uint32_t class_idx,
                           float weight);

/**
 * Predict consensus class from hard model votes.
 *
 * @param f              Fusion instance
 * @param model_votes    Array of n_models class indices (hard votes)
 * @param similarities   Output: similarity scores (nullable, n_classes elements)
 * @return               Consensus class index
 */
uint32_t isildur_fusion_predict_hard(const isildur_fusion_t *f,
                                      const uint32_t *model_votes,
                                      float *similarities);

/**
 * Predict consensus class from soft model outputs.
 *
 * @param f             Fusion instance
 * @param probabilities Flat array [n_models * n_classes] row-major
 * @param similarities  Output: similarity scores (nullable, n_classes)
 * @return              Consensus class index
 */
uint32_t isildur_fusion_predict_soft(const isildur_fusion_t *f,
                                      const float *probabilities,
                                      float *similarities);

/**
 * Remove a model's contributions from the consensus.
 */
void isildur_fusion_remove_model(isildur_fusion_t *f, uint32_t model_idx);

/**
 * Estimate a model's contribution to the consensus (∈ [-1, 1]).
 */
float isildur_fusion_model_contribution(const isildur_fusion_t *f,
                                         uint32_t model_idx);

/**
 * Normalize consensus memory (call after all training).
 */
void isildur_fusion_normalize(isildur_fusion_t *f);

/* ── HDC Encoder (SpikeHDC + Associative Memory) ────────────────── */

typedef struct {
    isildur_itemmem_t  *item_mem;    /** Item memory for scalar quantization */
    isildur_hv_t       **keys;       /** Key hypervectors [input_size] */
    isildur_hv_t       **class_hvs;  /** Class prototype HVs [n_classes] */
    uint32_t            *class_counts;/** Samples per class [n_classes] */
    uint32_t             input_size;
    uint32_t             n_classes;
    uint32_t             dim;
} isildur_encoder_t;

/**
 * Create an HDC encoder (SpikeHDC + one-shot associative memory).
 *
 * @param input_size  Number of input features
 * @param n_classes   Number of output classes
 * @param dim         Hypervector dimension
 * @param n_levels    Item memory quantization levels
 * @param seed        Random seed
 * @return            Allocated encoder (caller must free)
 */
isildur_encoder_t *isildur_encoder_create(uint32_t input_size,
                                            uint32_t n_classes,
                                            uint32_t dim,
                                            uint32_t n_levels,
                                            uint64_t seed);

/** Free an encoder. */
void isildur_encoder_free(isildur_encoder_t *enc);

/**
 * Encode an input vector into a hypervector.
 *
 * @param enc     Encoder
 * @param inputs  Float array [input_size]
 * @return        Newly allocated hypervector (caller must free)
 */
isildur_hv_t *isildur_encoder_encode(const isildur_encoder_t *enc,
                                       const float *inputs);

/**
 * Train: add one encoded sample to a class.
 *
 * @param enc       Encoder
 * @param inputs    Float array [input_size]
 * @param class_idx Class label
 */
void isildur_encoder_train(isildur_encoder_t *enc,
                            const float *inputs,
                            uint32_t class_idx);

/**
 * Predict class for an input vector.
 *
 * @param enc    Encoder
 * @param inputs Float array [input_size]
 * @param dist   Output: Hamming distance to closest class (nullable)
 * @return       Predicted class index
 */
uint32_t isildur_encoder_predict(const isildur_encoder_t *enc,
                                   const float *inputs,
                                   uint32_t *dist);

/**
 * Finalize training: threshold class hypervectors.
 */
void isildur_encoder_finalize(isildur_encoder_t *enc);

/**
 * Predict with all similarity scores.
 *
 * @param enc          Encoder
 * @param inputs       Float array [input_size]
 * @param similarities Output: similarity scores [n_classes] (cosine * 1000)
 * @return             Predicted class index
 */
uint32_t isildur_encoder_predict_scores(const isildur_encoder_t *enc,
                                          const float *inputs,
                                          int32_t *similarities);

#ifdef __cplusplus
}
#endif

#endif /* ISILDUR_H */