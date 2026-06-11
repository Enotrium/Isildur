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

/* ── Lifecycle ──────────────────────────────────────────────────── */

/**
 * Allocate and initialize a hypervector.
 *
 * @param dim  Dimension in bits (e.g., 10000)
 * @return     Allocated hypervector (caller must free_hv)
 */
isildur_hv_t *isildur_alloc_hv(uint32_t dim);

/** Free a hypervector. */
void isildur_free_hv(isildur_hv_t *hv);

/* ── Generation ─────────────────────────────────────────────────── */

/**
 * Generate a random bipolar hypervector (balanced coin flips).
 *
 * Uses a simple LCG with a user-provided seed for determinism.
 * The resulting vector is *approximately* balanced (±1 count ~equal).
 * For exact balance, call isildur_balance() after generation.
 *
 * @param hv    Pre-allocated hypervector
 * @param seed  Random seed (0 = use time-based seed)
 */
void isildur_gen_hv(isildur_hv_t *hv, uint64_t seed);

/**
 * Generate a balanced bipolar hypervector (exactly 50/50 ±1).
 */
void isildur_gen_balanced_hv(isildur_hv_t *hv, uint64_t seed);

/**
 * Balance an existing hypervector to exactly 50% +1 / 50% -1.
 * Modifies in place.
 */
void isildur_balance(isildur_hv_t *hv);

/* ── Core HDC Operations ────────────────────────────────────────── */

/**
 * Bind two hypervectors: element-wise XOR.
 *
 * In bipolar {-1,+1}, XOR is equivalent to multiplication:
 *   bind(a, b) = a XOR b
 *
 * Hardware: d XOR gates, single cycle.
 * C: d/64 XOR operations on uint64_t words.
 *
 * @param result  Output: result = a XOR b
 * @param a       First hypervector
 * @param b       Second hypervector
 */
void isildur_bind(isildur_hv_t *result,
                  const isildur_hv_t *a,
                  const isildur_hv_t *b);

/**
 * Unbind: recover value from key-value binding.
 * bind(bind(v, k), k) = v. XOR is its own inverse.
 */
void isildur_unbind(isildur_hv_t *result,
                    const isildur_hv_t *bound,
                    const isildur_hv_t *key);

/**
 * Bundle (superpose) k hypervectors: accumulate + sign-threshold.
 *
 * For each bit position:
 *   sum = count of 1 bits across all k vectors
 *   result bit = 1 if sum > k/2, else 0
 *
 * This is the HDC majority-vote operation.
 *
 * @param result  Output: bundled hypervector
 * @param hvs     Array of k hypervectors
 * @param k       Number of vectors to bundle
 */
void isildur_bundle(isildur_hv_t *result,
                    const isildur_hv_t **hvs,
                    uint32_t k);

/**
 * Permute: cyclic right-shift by k positions.
 *
 * In C: rotate the packed bit array right by k.
 * Hardware: barrel shifter, single cycle.
 */
void isildur_permute(isildur_hv_t *result,
                     const isildur_hv_t *hv,
                     uint32_t shift);

/* ── Similarity ─────────────────────────────────────────────────── */

/**
 * Compute Hamming distance between two hypervectors.
 *
 * d_H(a, b) = #{i : a_i != b_i}
 *
 * For bipolar vectors: XOR + popcount.
 * Hardware: XOR + reduction tree, O(log d) cycles.
 *
 * @return  Hamming distance (0..dim)
 */
uint32_t isildur_hamming(const isildur_hv_t *a,
                          const isildur_hv_t *b);

/**
 * Compute cosine similarity between two hypervectors.
 *
 * sim(a, b) = (a·b) / dim = (dim - 2*d_H) / dim
 * Range: [-1, 1]. 1 = identical, 0 = random, -1 = opposite.
 *
 * @return  Cosine similarity in fixed-point: value * 1000 (e.g., 1000 = 1.0)
 */
int32_t isildur_similarity(const isildur_hv_t *a,
                            const isildur_hv_t *b);

/* ── Popcount ───────────────────────────────────────────────────── */

/**
 * Count set bits (popcount / Hamming weight) of a hypervector.
 *
 * @return  Number of bits set to 1 (0..dim)
 */
uint32_t isildur_popcount(const isildur_hv_t *hv);

/* ── Associative Memory ─────────────────────────────────────────── */

/**
 * Associative memory: find the class with minimum Hamming distance.
 *
 * This is the core HDC inference operation:
 *   1. Compute Hamming(query, class[i]) for all i
 *   2. Return the index with minimum distance
 *
 * Energy (CIM TCAM, 28nm): ~0.3 pJ per inference
 * Time: O(n_classes * d/64) word operations
 *
 * @param query        Query hypervector
 * @param class_hvs    Array of n_classes prototype hypervectors
 * @param n_classes    Number of classes
 * @param distances    Output: Hamming distances to all classes (nullable)
 * @return             Index of closest class (0..n_classes-1)
 */
uint32_t isildur_assoc_infer(const isildur_hv_t *query,
                              const isildur_hv_t **class_hvs,
                              uint32_t n_classes,
                              uint32_t *distances);

/**
 * Associative memory inference with top-k results.
 *
 * @param query        Query hypervector
 * @param class_hvs    Array of n_classes prototype hypervectors
 * @param n_classes    Number of classes
 * @param k            Number of top results
 * @param top_indices  Output: indices of top k closest classes
 * @param top_dists    Output: distances of top k closest classes
 */
void isildur_assoc_topk(const isildur_hv_t *query,
                         const isildur_hv_t **class_hvs,
                         uint32_t n_classes,
                         uint32_t k,
                         uint32_t *top_indices,
                         uint32_t *top_dists);

/**
 * Train associative memory: add one sample to a class prototype.
 *
 * Incremental learning: class_hv[class_id] = bundle(class_hv[class_id], sample_hv)
 *
 * @param class_hv  Class prototype hypervector (modified in place)
 * @param sample_hv Training sample hypervector
 * @param count     Current sample count for this class (updated)
 */
void isildur_assoc_train(isildur_hv_t *class_hv,
                          const isildur_hv_t *sample_hv,
                          uint32_t *count);

/* ── Utility ────────────────────────────────────────────────────── */

/**
 * Copy hypervector a to b (b = a).
 */
void isildur_copy(isildur_hv_t *dst, const isildur_hv_t *src);

/**
 * Compare two hypervectors for equality.
 */
bool isildur_equals(const isildur_hv_t *a, const isildur_hv_t *b);

/**
 * Serialize a hypervector to packed bytes.
 *
 * @param hv       Hypervector to serialize
 * @param buf      Output buffer (must be ISILDUR_WORDS(dim) * 8 bytes)
 * @param buf_size Size of output buffer
 * @return         Number of bytes written
 */
size_t isildur_serialize(const isildur_hv_t *hv,
                           uint8_t *buf, size_t buf_size);

/**
 * Deserialize a hypervector from packed bytes.
 */
isildur_hv_t *isildur_deserialize(const uint8_t *buf,
                                    size_t buf_size, uint32_t dim);

/**
 * Print a hypervector (first n bits for debugging).
 */
void isildur_print(const isildur_hv_t *hv, uint32_t max_bits);

#ifdef __cplusplus
}
#endif

#endif /* ISILDUR_H */