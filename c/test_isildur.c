/**
 * test_isildur.c — Comprehensive test suite for Isildur C library.
 *
 * Tests all submodules:
 *   - Core HDC ops: bind, bundle, unbind, permute, hamming, similarity, popcount
 *   - RNG, alloc/clone, copy, equals, serialize/deserialize, print
 *   - Associative memory: infer, topk, train
 *   - Noise/corruption
 *   - Item memory: scalar and vector encoding
 *   - CIM: hamming distance, predict, batch predict, energy estimate
 *   - Fusion: hard/soft predict, training, model removal, contribution, normalize
 *   - Encoder: encode, train, predict, finalize, predict scores
 *
 * All tests use assert() for verification. Returns 0 on success.
 */
#include "isildur.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define DIM 10000
#define SMALL_DIM 256

static int tests_passed = 0;
static int tests_failed = 0;
static int test_num = 0;

#define TEST(name) do { \
    test_num++; \
    printf("  TEST %2d: %s ... ", test_num, name); \
    fflush(stdout); \
} while (0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while (0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    tests_failed++; \
} while (0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while (0)

/* ────────────────────────────────────────────────────────────────
 * Helper: free array of HVs
 * ──────────────────────────────────────────────────────────────── */
static void free_hvs(isildur_hv_t **hvs, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) isildur_free_hv(hvs[i]);
    free(hvs);
}

/* ────────────────────────────────────────────────────────────────
 * Test: RNG
 * ──────────────────────────────────────────────────────────────── */
static void test_rng(void) {
    TEST("RNG init and generate");
    isildur_rng_t rng;
    isildur_rng_init(&rng, 42);
    uint64_t a = isildur_rng_next(&rng);
    uint64_t b = isildur_rng_next(&rng);
    CHECK(a != b, "RNG values should differ");
    CHECK(a != 0, "RNG should produce non-zero");

    /* Deterministic: same seed → same sequence */
    isildur_rng_init(&rng, 42);
    uint64_t a2 = isildur_rng_next(&rng);
    CHECK(a == a2, "RNG must be deterministic");

    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: alloc, clone, copy, equals
 * ──────────────────────────────────────────────────────────────── */
static void test_lifecycle(void) {
    TEST("Alloc/Free");
    isildur_hv_t *hv = isildur_alloc_hv(SMALL_DIM);
    CHECK(hv != NULL, "alloc_hv must succeed");
    CHECK(hv->dim == SMALL_DIM, "dimension mismatch");
    CHECK(hv->n_words == ISILDUR_WORDS(SMALL_DIM), "n_words mismatch");
    isildur_free_hv(hv);
    PASS();

    TEST("Clone");
    isildur_hv_t *a = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(a, 1);
    isildur_hv_t *b = isildur_clone_hv(a);
    CHECK(b != NULL, "clone must succeed");
    CHECK(isildur_equals(a, b), "clone must be equal");
    CHECK(a != b, "clone must be different pointer");
    isildur_free_hv(a);
    isildur_free_hv(b);
    PASS();

    TEST("Copy and Equals");
    a = isildur_alloc_hv(SMALL_DIM);
    b = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(a, 1);
    isildur_gen_hv(b, 2);
    CHECK(!isildur_equals(a, b), "different HVs should not be equal");
    isildur_copy(b, a);
    CHECK(isildur_equals(a, b), "copied HVs must be equal");
    isildur_free_hv(a);
    isildur_free_hv(b);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: generation, balance, popcount
 * ──────────────────────────────────────────────────────────────── */
static void test_generation(void) {
    TEST("gen_hv and popcount");
    isildur_hv_t *hv = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(hv, 42);
    uint32_t pc = isildur_popcount(hv);
    CHECK(pc > 0 && pc < SMALL_DIM, "popcount should be > 0 and < dim");
    /* For 256 bits, random roughly 50/50 */
    CHECK(pc >= 80 && pc <= 176, "popcount should be roughly balanced");
    isildur_free_hv(hv);
    PASS();

    TEST("gen_balanced_hv");
    hv = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(hv, 42);
    pc = isildur_popcount(hv);
    /* For even 256: exactly 128 ones */
    CHECK(pc == SMALL_DIM / 2, "balanced HV must have exactly dim/2 ones");
    isildur_free_hv(hv);
    PASS();

    TEST("balance");
    hv = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(hv, 43);
    isildur_balance(hv);
    pc = isildur_popcount(hv);
    CHECK(pc == SMALL_DIM / 2, "balance must produce exactly dim/2 ones");
    isildur_free_hv(hv);
    PASS();

    TEST("gen_hv_batch");
    uint32_t n = 10;
    isildur_hv_t **hvs = calloc(n, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n; i++) hvs[i] = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv_batch(hvs, n, 100);
    /* Verify all are different */
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i + 1; j < n; j++) {
            CHECK(!isildur_equals(hvs[i], hvs[j]), "batch HVs should all differ");
        }
    }
    free_hvs(hvs, n);
    PASS();

    TEST("gen_hv_rng");
    isildur_rng_t rng;
    isildur_rng_init(&rng, 77);
    isildur_hv_t *a = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t *b = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv_rng(a, &rng);
    isildur_rng_init(&rng, 77);
    isildur_gen_hv_rng(b, &rng);
    CHECK(isildur_equals(a, b), "RNG-based generation must be deterministic");
    isildur_free_hv(a);
    isildur_free_hv(b);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: core HDC operations
 * ──────────────────────────────────────────────────────────────── */
static void test_core_ops(void) {
    TEST("bind/unbind (XOR self-inverse)");
    isildur_hv_t *a = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t *b = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t *r = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t *recovered = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(a, 1);
    isildur_gen_hv(b, 2);
    isildur_bind(r, a, b);
    isildur_unbind(recovered, r, b);
    CHECK(isildur_equals(a, recovered), "unbind should recover original");
    isildur_free_hv(a); isildur_free_hv(b);
    isildur_free_hv(r); isildur_free_hv(recovered);
    PASS();

    TEST("bundle (majority vote)");
    uint32_t k = 3;
    isildur_hv_t **hvs = calloc(k, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < k; i++) {
        hvs[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_hv(hvs[i], 10 + i);
    }
    isildur_hv_t *bundled = isildur_alloc_hv(SMALL_DIM);
    isildur_bundle(bundled, (const isildur_hv_t **)hvs, k);
    /* Bundled HV should have dim bits and not be zero */
    uint32_t pc = isildur_popcount(bundled);
    CHECK(pc > 0, "bundle should not be all zeros");
    isildur_free_hv(bundled);
    free_hvs(hvs, k);
    PASS();

    TEST("bundle empty");
    isildur_hv_t *empty = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t **no_hvs = NULL;
    isildur_bundle(empty, (const isildur_hv_t **)no_hvs, 0);
    uint32_t pc_empty = isildur_popcount(empty);
    CHECK(pc_empty == 0, "bundle of 0 vectors should be all zeros");
    isildur_free_hv(empty);
    PASS();

    TEST("bundle_int32 accumulate and finalize");
    isildur_hv_t *hv = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(hv, 42);
    int32_t *accums = calloc(SMALL_DIM, sizeof(int32_t));
    uint32_t count = 0;
    isildur_bundle_int32_accumulate(accums, hv, SMALL_DIM, &count);
    CHECK(count == 1, "count should be 1 after one accumulate");
    isildur_hv_t *result = isildur_alloc_hv(SMALL_DIM);
    isildur_bundle_int32_finalize(result, accums, SMALL_DIM, count);
    /* After one vector, result should duplicate input */
    CHECK(isildur_equals(result, hv), "finalize from 1 vector should match input");
    free(accums);
    isildur_free_hv(hv);
    isildur_free_hv(result);
    PASS();

    TEST("permute (cyclic shift)");
    isildur_hv_t *src = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t *dst = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(src, 1);
    isildur_permute(dst, src, 0);
    CHECK(isildur_equals(src, dst), "permute(0) should be identity");
    isildur_permute(dst, src, SMALL_DIM);
    CHECK(isildur_equals(src, dst), "permute(dim) should be identity");
    /* Permute by 1 should change vector */
    isildur_permute(dst, src, 1);
    CHECK(!isildur_equals(src, dst), "permute(1) should change vector");
    isildur_free_hv(src); isildur_free_hv(dst);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: similarity and distance
 * ──────────────────────────────────────────────────────────────── */
static void test_similarity(void) {
    TEST("hamming distance");
    isildur_hv_t *a = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t *b = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_hv(a, 1);
    isildur_gen_hv(b, 2);
    uint32_t d = isildur_hamming(a, b);
    CHECK(d > 0, "different HVs should have positive hamming");
    CHECK(d <= SMALL_DIM, "hamming should not exceed dim");
    d = isildur_hamming(a, a);
    CHECK(d == 0, "self hamming must be 0");
    isildur_free_hv(a); isildur_free_hv(b);
    PASS();

    TEST("similarity (cosine * 1000)");
    a = isildur_alloc_hv(SMALL_DIM);
    b = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(a, 1);
    isildur_gen_balanced_hv(b, 2);
    int32_t sim = isildur_similarity(a, b);
    /* Two random balanced vectors: cosine ~0 (± ~0.06 for d=256 = 1/16)
     * Bigger tolerance for small dim */
    CHECK(sim > -500 && sim < 500, "random balanced vectors should be ~0 similarity");
    sim = isildur_similarity(a, a);
    CHECK(sim == 1000, "self-similarity must be 1000");
    isildur_free_hv(a); isildur_free_hv(b);
    PASS();

    TEST("hamming_batch");
    uint32_t n = 5;
    isildur_hv_t *query = isildur_alloc_hv(SMALL_DIM);
    isildur_hv_t **cands = calloc(n, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n; i++) {
        cands[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_hv(cands[i], 100 + i);
    }
    isildur_gen_hv(query, 99);
    uint32_t *dists = calloc(n, sizeof(uint32_t));
    isildur_hamming_batch(query, (const isildur_hv_t **)cands, n, dists);
    for (uint32_t i = 0; i < n; i++) {
        CHECK(dists[i] == isildur_hamming(query, cands[i]), "batch hamming must match pairwise");
    }
    free(dists);
    isildur_free_hv(query);
    free_hvs(cands, n);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: associative memory
 * ──────────────────────────────────────────────────────────────── */
static void test_assoc_memory(void) {
    TEST("assoc_infer");
    uint32_t n_classes = 5;
    isildur_hv_t **classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
        classes[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(classes[i], 1000 + i);
    }
    isildur_hv_t *query = isildur_clone_hv(classes[2]);
    /* Corrupt slightly to simulate noisy query */
    isildur_rng_t rng;
    isildur_rng_init(&rng, 999);
    isildur_corrupt_hv(query, query, 0.05f, ISILDUR_NOISE_FLIP, &rng);

    uint32_t *dists = calloc(n_classes, sizeof(uint32_t));
    uint32_t pred = isildur_assoc_infer(query, (const isildur_hv_t **)classes, n_classes, dists);
    CHECK(pred == 2, "should correctly identify class 2 despite noise");

    /* Distance to class 2 should be smallest */
    for (uint32_t i = 0; i < n_classes; i++) {
        CHECK(dists[pred] <= dists[i], "predicted class should have min distance");
    }
    free(dists);
    isildur_free_hv(query);
    free_hvs(classes, n_classes);
    PASS();

    TEST("assoc_topk");
    n_classes = 5;
    classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
        classes[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(classes[i], 2000 + i);
    }
    query = isildur_clone_hv(classes[0]);
    uint32_t k = 3;
    uint32_t *ti = calloc(k, sizeof(uint32_t));
    uint32_t *td = calloc(k, sizeof(uint32_t));
    isildur_assoc_topk(query, (const isildur_hv_t **)classes, n_classes, k, ti, td);
    CHECK(ti[0] == 0, "top-1 should be class 0");
    CHECK(td[0] == 0, "distance to self should be 0");
    for (uint32_t i = 0; i < k - 1; i++) {
        CHECK(td[i] <= td[i + 1], "topk should be sorted by distance");
    }
    free(ti); free(td);
    isildur_free_hv(query);
    free_hvs(classes, n_classes);
    PASS();

    TEST("assoc_train (incremental learning)");
    isildur_hv_t *class_hv = isildur_alloc_hv(SMALL_DIM);
    memset(class_hv->bits, 0, class_hv->n_words * 8);
    uint32_t count = 0;
    isildur_hv_t *sample = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(sample, 1);
    isildur_assoc_train(class_hv, sample, &count);
    CHECK(count == 1, "count should be 1 after training");
    /* After one sample, class_hv should equal sample */
    CHECK(isildur_equals(class_hv, sample), "single-sample training should produce identity");
    isildur_free_hv(class_hv);
    isildur_free_hv(sample);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: serialize / deserialize
 * ──────────────────────────────────────────────────────────────── */
static void test_serialize(void) {
    TEST("serialize and deserialize");
    isildur_hv_t *hv = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(hv, 42);
    size_t sz = hv->n_words * 8 + 4;
    uint8_t *buf = calloc(sz, 1);
    size_t written = isildur_serialize(hv, buf, sz);
    CHECK(written == sz, "serialize should write expected bytes");
    isildur_hv_t *restored = isildur_deserialize(buf, sz, SMALL_DIM);
    CHECK(restored != NULL, "deserialize must succeed");
    CHECK(isildur_equals(hv, restored), "deserialized must match original");
    isildur_free_hv(hv);
    isildur_free_hv(restored);
    free(buf);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: noise / corruption
 * ──────────────────────────────────────────────────────────────── */
static void test_noise(void) {
    TEST("corrupt_hv - flip");
    isildur_hv_t *orig = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(orig, 1);
    isildur_hv_t *corrupt = isildur_alloc_hv(SMALL_DIM);
    isildur_rng_t rng;
    isildur_rng_init(&rng, 42);
    isildur_corrupt_hv(corrupt, orig, 0.5f, ISILDUR_NOISE_FLIP, &rng);
    /* At 50% flip rate, roughly half should differ */
    uint32_t d = isildur_hamming(orig, corrupt);
    CHECK(d >= SMALL_DIM / 4 && d <= 3 * SMALL_DIM / 4, "50%% flip should change roughly half");
    isildur_free_hv(orig);
    isildur_free_hv(corrupt);
    PASS();

    TEST("corrupt_hv - drop");
    isildur_hv_t *a = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(a, 1);
    isildur_hv_t *b = isildur_alloc_hv(SMALL_DIM);
    isildur_rng_init(&rng, 777);
    isildur_corrupt_hv(b, a, 0.3f, ISILDUR_NOISE_DROP, &rng);
    /* Dropping zeros out bits; popcount should decrease */
    uint32_t pc_a = isildur_popcount(a);
    uint32_t pc_b = isildur_popcount(b);
    CHECK(pc_b < pc_a, "dropping should reduce popcount");
    isildur_free_hv(a);
    isildur_free_hv(b);
    PASS();

    TEST("corrupt_hv - in-place");
    isildur_hv_t *hv = isildur_alloc_hv(SMALL_DIM);
    isildur_gen_balanced_hv(hv, 1);
    isildur_rng_init(&rng, 42);
    isildur_corrupt_hv(hv, hv, 0.1f, ISILDUR_NOISE_FLIP, &rng);
    /* Should still be valid (not segfault, not all zeros) */
    uint32_t pc = isildur_popcount(hv);
    CHECK(pc > 0, "in-place corruption must produce valid result");
    isildur_free_hv(hv);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: Item Memory
 * ──────────────────────────────────────────────────────────────── */
static void test_item_memory(void) {
    TEST("itemmem_create and get_level");
    uint32_t n_levels = 5;
    isildur_itemmem_t *im = isildur_itemmem_create(n_levels, SMALL_DIM, 42, 0.1f);
    CHECK(im != NULL, "item memory creation must succeed");
    CHECK(im->n_levels == n_levels, "n_levels mismatch");
    for (uint32_t i = 0; i < n_levels; i++) {
        const isildur_hv_t *lv = isildur_itemmem_get_level(im, i);
        CHECK(lv != NULL, "each level must exist");
        CHECK(lv->dim == SMALL_DIM, "level dim mismatch");
    }
    isildur_itemmem_free(im);
    PASS();

    TEST("itemmem_encode_scalar");
    im = isildur_itemmem_create(10, SMALL_DIM, 42, 0.05f);
    isildur_hv_t *low = isildur_itemmem_encode_scalar(im, 0.0f, 0.0f, 1.0f);
    isildur_hv_t *mid = isildur_itemmem_encode_scalar(im, 0.5f, 0.0f, 1.0f);
    isildur_hv_t *high = isildur_itemmem_encode_scalar(im, 1.0f, 0.0f, 1.0f);
    CHECK(low != NULL && mid != NULL && high != NULL, "encode_scalar must succeed");
    /* low and high should differ more than low and mid */
    uint32_t d_low_mid = isildur_hamming(low, mid);
    uint32_t d_low_high = isildur_hamming(low, high);
    CHECK(d_low_high > d_low_mid, "distant values should have higher hamming distance");
    isildur_free_hv(low); isildur_free_hv(mid); isildur_free_hv(high);
    isildur_itemmem_free(im);
    PASS();

    TEST("itemmem_encode_scalar clamping");
    im = isildur_itemmem_create(5, SMALL_DIM, 42, 0.1f);
    /* Below range should clamp to level 0 */
    isildur_hv_t *below = isildur_itemmem_encode_scalar(im, -1.0f, 0.0f, 1.0f);
    isildur_hv_t *zero = isildur_itemmem_encode_scalar(im, 0.0f, 0.0f, 1.0f);
    CHECK(isildur_hamming(below, zero) == 0, "clamped low values should match min level");
    isildur_free_hv(below); isildur_free_hv(zero);

    /* Above range should clamp to level n_levels-1 */
    isildur_hv_t *above = isildur_itemmem_encode_scalar(im, 10.0f, 0.0f, 1.0f);
    isildur_hv_t *one = isildur_itemmem_encode_scalar(im, 1.0f, 0.0f, 1.0f);
    CHECK(isildur_hamming(above, one) == 0, "clamped high values should match max level");
    isildur_free_hv(above); isildur_free_hv(one);
    isildur_itemmem_free(im);
    PASS();

    TEST("itemmem_encode_vector");
    im = isildur_itemmem_create(10, SMALL_DIM, 42, 0.05f);
    uint32_t n_values = 4;
    isildur_hv_t **keys = calloc(n_values, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_values; i++) {
        keys[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(keys[i], 5000 + i);
    }
    float values[] = {0.1f, 0.3f, 0.5f, 0.9f};
    isildur_hv_t *encoded = isildur_itemmem_encode_vector(im, values, n_values, keys, 0.0f, 1.0f);
    CHECK(encoded != NULL, "encode_vector must succeed");
    /* Encoded result should be a valid bundled HV */
    uint32_t pc = isildur_popcount(encoded);
    CHECK(pc > 0, "encoded vector should not be all zeros");
    isildur_free_hv(encoded);
    free_hvs(keys, n_values);
    isildur_itemmem_free(im);
    PASS();

    TEST("itemmem_encode_vector auto-range");
    im = isildur_itemmem_create(5, SMALL_DIM, 42, 0.1f);
    n_values = 3;
    keys = calloc(n_values, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_values; i++) {
        keys[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(keys[i], 6000 + i);
    }
    float auto_values[] = {0.0f, 0.0f, 0.0f};
    /* When min==0 and max==0, auto-detect */
    isildur_hv_t *result = isildur_itemmem_encode_vector(im, auto_values, n_values, keys, 0.0f, 0.0f);
    CHECK(result != NULL, "auto-range encoding must succeed");
    isildur_free_hv(result);
    free_hvs(keys, n_values);
    isildur_itemmem_free(im);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: CIM Hamming distance
 * ──────────────────────────────────────────────────────────────── */
static void test_cim(void) {
    TEST("cim_create and free");
    isildur_cim_config_t config = {0};
    config.block_size = 32;
    config.process_variation = 0.0f;
    isildur_cim_t *cim = isildur_cim_create(&config);
    CHECK(cim != NULL, "cim_create must succeed");
    isildur_cim_free(cim);
    PASS();

    TEST("cim_create default config");
    isildur_cim_t *cim2 = isildur_cim_create(NULL);
    CHECK(cim2 != NULL, "cim_create with NULL config must succeed");
    CHECK(cim2->config.block_size == 32, "default block_size should be 32");
    isildur_cim_free(cim2);
    PASS();

    TEST("cim_load and hamming");
    cim = isildur_cim_create(&config);
    uint32_t n_classes = 3;
    isildur_hv_t **classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
        classes[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(classes[i], 1000 + i);
    }
    isildur_cim_load(cim, classes, n_classes);
    CHECK(cim->config.n_classes == n_classes, "load should set n_classes");

    /* Query: exact copy of class 0 */
    isildur_hv_t *query = isildur_clone_hv(classes[0]);
    uint32_t *dists = calloc(n_classes, sizeof(uint32_t));
    isildur_cim_hamming(cim, query, dists);
    CHECK(dists[0] == 0, "exact match should have hamming=0");
    isildur_free_hv(query);
    free(dists);
    free_hvs(classes, n_classes);
    isildur_cim_free(cim);
    PASS();

    TEST("cim_predict");
    config.block_size = 16;
    cim = isildur_cim_create(&config);
    n_classes = 4;
    classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
        classes[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(classes[i], 2000 + i);
    }
    isildur_cim_load(cim, classes, n_classes);

    query = isildur_clone_hv(classes[3]);
    uint32_t dist;
    uint32_t pred = isildur_cim_predict(cim, query, &dist);
    CHECK(pred == 3, "cim_predict should return correct class");
    CHECK(dist == 0, "exact match should have distance=0");

    isildur_free_hv(query);
    free_hvs(classes, n_classes);
    isildur_cim_free(cim);
    PASS();

    TEST("cim_predict_batch");
    config.block_size = 16;
    cim = isildur_cim_create(&config);
    n_classes = 3;
    classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
        classes[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(classes[i], 3000 + i);
    }
    isildur_cim_load(cim, classes, n_classes);

    uint32_t n_queries = 3;
    isildur_hv_t **queries = calloc(n_queries, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_queries; i++) {
        queries[i] = isildur_clone_hv(classes[i]);
    }
    uint32_t *preds = calloc(n_queries, sizeof(uint32_t));
    uint32_t *batch_dists = calloc(n_queries, sizeof(uint32_t));
    isildur_cim_predict_batch(cim, queries, n_queries, preds, batch_dists);
    for (uint32_t i = 0; i < n_queries; i++) {
        CHECK(preds[i] == i, "batch prediction must be correct for each query");
        CHECK(batch_dists[i] == 0, "exact matches should have 0 distance");
    }
    free(preds); free(batch_dists);
    free_hvs(queries, n_queries);
    free_hvs(classes, n_classes);
    isildur_cim_free(cim);
    PASS();

    TEST("cim_energy_estimate");
    config.block_size = 32;
    config.n_classes = 10;
    config.hv_dim = 10000;
    config.n_blocks = (10000 + 31) / 32;
    cim = isildur_cim_create(&config);
    float energy = isildur_cim_energy_pj(cim);
    CHECK(energy > 0.0f, "energy estimate should be positive");
    isildur_cim_free(cim);
    PASS();

    TEST("cim with process variation");
    config.block_size = 16;
    config.process_variation = 0.1f;
    cim = isildur_cim_create(&config);
    n_classes = 3;
    classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
        classes[i] = isildur_alloc_hv(SMALL_DIM);
        isildur_gen_balanced_hv(classes[i], 4000 + i);
    }
    isildur_cim_load(cim, classes, n_classes);
    query = isildur_clone_hv(classes[0]);
    dists = calloc(n_classes, sizeof(uint32_t));
    isildur_cim_hamming(cim, query, dists);
    /* With variation, may not be exactly 0 but should be small */
    CHECK(dists[0] <= 1, "with variation, should be at most 1 off for exact match");
    free(dists);
    isildur_free_hv(query);
    free_hvs(classes, n_classes);
    isildur_cim_free(cim);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: HD-Glue Fusion
 * ──────────────────────────────────────────────────────────────── */
static void test_fusion(void) {
    TEST("fusion create and free");
    isildur_fusion_t *f = isildur_fusion_create(3, 5, SMALL_DIM, 42);
    CHECK(f != NULL, "fusion_create must succeed");
    CHECK(f->n_models == 3, "n_models mismatch");
    CHECK(f->n_classes == 5, "n_classes mismatch");
    isildur_fusion_free(f);
    PASS();

    TEST("fusion train and predict hard");
    f = isildur_fusion_create(3, 4, SMALL_DIM, 42);
    /* Train: all models agree on specific classes */
    isildur_fusion_train(f, 0, 0, 1.0f);
    isildur_fusion_train(f, 1, 0, 1.0f);
    isildur_fusion_train(f, 2, 0, 1.0f);
    isildur_fusion_train(f, 0, 1, 1.0f);
    isildur_fusion_train(f, 1, 2, 1.0f);
    isildur_fusion_train(f, 2, 3, 1.0f);

    /* All models voting for class 0 */
    uint32_t votes[] = {0, 0, 0};
    float sims[4];
    uint32_t pred = isildur_fusion_predict_hard(f, votes, sims);
    CHECK(pred == 0, "unanimous vote should produce consensus class 0");
    (void)sims; /* similarity scores verified only through prediction correctness */
    isildur_fusion_free(f);
    PASS();

    TEST("fusion predict soft");
    f = isildur_fusion_create(2, 3, SMALL_DIM, 42);
    isildur_fusion_train(f, 0, 0, 1.0f);
    isildur_fusion_train(f, 1, 0, 1.0f);
    isildur_fusion_train(f, 0, 1, 0.5f);
    isildur_fusion_train(f, 1, 2, 0.5f);

    /* Soft probabilities: model 0 favors class 0, model 1 favors class 0 */
    float probs[] = {
        0.8f, 0.1f, 0.1f,  /* model 0 */
        0.7f, 0.2f, 0.1f   /* model 1 */
    };
    float soft_sims[3];
    uint32_t soft_pred = isildur_fusion_predict_soft(f, probs, soft_sims);
    CHECK(soft_pred == 0, "soft inference should favor class 0");
    isildur_fusion_free(f);
    PASS();

    TEST("fusion remove model");
    f = isildur_fusion_create(3, 3, SMALL_DIM, 42);
    /* Train with model 0 for class 0 */
    isildur_fusion_train(f, 0, 0, 1.0f);
    isildur_fusion_train(f, 1, 1, 1.0f);
    isildur_fusion_train(f, 2, 2, 1.0f);

    /* Remove model 0 */
    isildur_fusion_remove_model(f, 0);

    /* After removal, model 0's influence should be reduced */
    /* Verify no crash and function succeeds */
    uint32_t votes2[] = {1, 1, 1};
    uint32_t pred2 = isildur_fusion_predict_hard(f, votes2, NULL);
    /* Should still produce valid class */
    CHECK(pred2 < f->n_classes, "removal should still produce valid result");
    isildur_fusion_free(f);
    PASS();

    TEST("fusion model contribution");
    f = isildur_fusion_create(2, 4, SMALL_DIM, 42);
    isildur_fusion_train(f, 0, 0, 1.0f);
    isildur_fusion_train(f, 0, 1, 1.0f);
    isildur_fusion_train(f, 1, 2, 1.0f);
    isildur_fusion_train(f, 1, 3, 1.0f);

    float contrib = isildur_fusion_model_contribution(f, 0);
    CHECK(contrib >= -1.0f && contrib <= 1.0f, "contribution should be in [-1, 1]");
    float contrib2 = isildur_fusion_model_contribution(f, 1);
    CHECK(contrib2 >= -1.0f && contrib2 <= 1.0f, "contribution should be in [-1, 1]");
    isildur_fusion_free(f);
    PASS();

    TEST("fusion normalize");
    f = isildur_fusion_create(2, 3, SMALL_DIM, 42);
    isildur_fusion_train(f, 0, 0, 1.0f);
    isildur_fusion_train(f, 1, 1, 1.0f);
    isildur_fusion_normalize(f);
    /* After normalize, memory_hv should be balanced */
    uint32_t pc = isildur_popcount(f->memory_hv);
    CHECK(pc == SMALL_DIM / 2, "normalized memory should be exactly balanced");
    isildur_fusion_free(f);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: HDC Encoder
 * ──────────────────────────────────────────────────────────────── */
static void test_encoder(void) {
    TEST("encoder create and free");
    isildur_encoder_t *enc = isildur_encoder_create(10, 5, SMALL_DIM, 8, 42);
    CHECK(enc != NULL, "encoder_create must succeed");
    CHECK(enc->input_size == 10, "input_size mismatch");
    CHECK(enc->n_classes == 5, "n_classes mismatch");
    isildur_encoder_free(enc);
    PASS();

    TEST("encoder encode");
    enc = isildur_encoder_create(5, 3, SMALL_DIM, 8, 42);
    float inputs[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    isildur_hv_t *hv = isildur_encoder_encode(enc, inputs);
    CHECK(hv != NULL, "encode must succeed");
    uint32_t pc = isildur_popcount(hv);
    CHECK(pc > 0, "encoded HV should not be all zeros");
    isildur_free_hv(hv);
    isildur_encoder_free(enc);
    PASS();

    TEST("encoder train and predict");
    enc = isildur_encoder_create(3, 3, SMALL_DIM, 5, 42);
    /* Train: 5 samples per class */
    for (uint32_t c = 0; c < 3; c++) {
        for (uint32_t s = 0; s < 5; s++) {
            float input[3];
            for (uint32_t i = 0; i < 3; i++) {
                input[i] = (float)(c * 10 + s + i) * 0.1f;
            }
            isildur_encoder_train(enc, input, c);
        }
    }

    /* Test prediction on training-like inputs */
    float test_input[] = {0.0f, 0.1f, 0.2f};
    uint32_t dist;
    uint32_t pred = isildur_encoder_predict(enc, test_input, &dist);
    CHECK(pred < enc->n_classes, "prediction must return valid class");
    CHECK(dist <= SMALL_DIM, "distance must be <= dim");
    isildur_encoder_free(enc);
    PASS();

    TEST("encoder finalize");
    enc = isildur_encoder_create(4, 3, SMALL_DIM, 5, 42);
    float input2[] = {0.1f, 0.2f, 0.3f, 0.4f};
    isildur_encoder_train(enc, input2, 0);
    isildur_encoder_finalize(enc);
    /* After finalize, class 0 should be balanced; classes 1,2 get random */
    uint32_t pc0 = isildur_popcount(enc->class_hvs[0]);
    CHECK(pc0 == SMALL_DIM / 2, "finalize should balance trained class");
    isildur_encoder_free(enc);
    PASS();

    TEST("encoder predict with scores");
    enc = isildur_encoder_create(3, 3, SMALL_DIM, 5, 42);
    for (uint32_t s = 0; s < 5; s++) {
        float in[3] = { (float)s * 0.1f, 0.0f, 0.0f };
        isildur_encoder_train(enc, in, 0);
    }
    float test_in[] = {0.0f, 0.0f, 0.0f};
    int32_t scores[3];
    uint32_t pred2 = isildur_encoder_predict_scores(enc, test_in, scores);
    CHECK(pred2 < enc->n_classes, "predict_scores must return valid class");
    /* Class 0 should have highest similarity score (only one trained) */
    CHECK(scores[pred2] >= scores[0] && scores[pred2] >= scores[1] && scores[pred2] >= scores[2],
          "winning class should have max score");
    isildur_encoder_free(enc);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Test: bit_set and bit_get
 * ──────────────────────────────────────────────────────────────── */
static void test_bit_access(void) {
    TEST("bit_set and bit_get");
    isildur_hv_t *hv = isildur_alloc_hv(SMALL_DIM);
    memset(hv->bits, 0, hv->n_words * 8);

    isildur_bit_set(hv, 0, 1);
    CHECK(isildur_bit_get(hv, 0) == 1, "bit 0 should be 1 after set");
    CHECK(isildur_bit_get(hv, 1) == 0, "bit 1 should still be 0");

    isildur_bit_set(hv, 200, 1);
    CHECK(isildur_bit_get(hv, 200) == 1, "bit 200 should be 1 after set");

    isildur_bit_set(hv, 200, 0);
    CHECK(isildur_bit_get(hv, 200) == 0, "bit 200 should be 0 after clear");

    /* Cross word boundary */
    isildur_bit_set(hv, 64, 1);
    CHECK(isildur_bit_get(hv, 64) == 1, "bit 64 (word boundary) should work");
    isildur_bit_set(hv, 64, 0);
    CHECK(isildur_bit_get(hv, 64) == 0, "clearing across word boundary should work");

    isildur_free_hv(hv);
    PASS();
}

/* ────────────────────────────────────────────────────────────────
 * Main
 * ──────────────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Isildur C Library Test Suite ===\n\n");

    printf("-- RNG --\n");
    test_rng();

    printf("\n-- Lifecycle --\n");
    test_lifecycle();

    printf("\n-- Generation --\n");
    test_generation();

    printf("\n-- Bit Access --\n");
    test_bit_access();

    printf("\n-- Core HDC Operations --\n");
    test_core_ops();

    printf("\n-- Similarity/Distance --\n");
    test_similarity();

    printf("\n-- Associative Memory --\n");
    test_assoc_memory();

    printf("\n-- Serialization --\n");
    test_serialize();

    printf("\n-- Noise/Corruption --\n");
    test_noise();

    printf("\n-- Item Memory --\n");
    test_item_memory();

    printf("\n-- CIM Hamming Distance --\n");
    test_cim();

    printf("\n-- Fusion (HD-Glue) --\n");
    test_fusion();

    printf("\n-- HDC Encoder --\n");
    test_encoder();

    printf("\n========================================\n");
    printf(" Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}