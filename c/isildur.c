/**
 * isildur.c — Isildur HDC/VSA Core Library (C Reference Implementation)
 * Portable C99. No dependencies beyond libc.
 */
#include "isildur.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* ────────────────────────────────────────────────────────────────
 * Population count lookup table (8-bit → popcount)
 * ──────────────────────────────────────────────────────────────── */
static const uint8_t pop8[256] = {
  0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
  1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
  1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
  2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
  1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
  2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
  2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
  3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

/* ────────────────────────────────────────────────────────────────
 * Popcount of a 64-bit word (bytewise lookup)
 * ──────────────────────────────────────────────────────────────── */
static inline uint32_t pc64(uint64_t x) {
  return (uint32_t)(pop8[x&0xFF]+pop8[(x>>8)&0xFF]+pop8[(x>>16)&0xFF]+
    pop8[(x>>24)&0xFF]+pop8[(x>>32)&0xFF]+pop8[(x>>40)&0xFF]+
    pop8[(x>>48)&0xFF]+pop8[(x>>56)&0xFF]);
}

/* ────────────────────────────────────────────────────────────────
 * Bit access helpers
 * ──────────────────────────────────────────────────────────────── */
static inline uint64_t bit_get_internal(const isildur_hv_t *hv, uint32_t i) {
  return (hv->bits[i/64] >> (i%64)) & 1ULL;
}
static inline void bit_set_internal(isildur_hv_t *hv, uint32_t i, uint64_t v) {
  uint32_t w=i/64,s=i%64; hv->bits[w]=(hv->bits[w]&~(1ULL<<s))|(v<<s);
}

/* ────────────────────────────────────────────────────────────────
 * Global LCG RNG (for backwards compatibility)
 * ──────────────────────────────────────────────────────────────── */
static uint64_t lcg_s=1;
static inline void lcg_seed(uint64_t s) {
  lcg_s=s?s:(uint64_t)time(NULL); if(!lcg_s)lcg_s=1;
}
static inline uint64_t lcg_next(void) {
  lcg_s=lcg_s*6364136223846793005ULL+1; return lcg_s;
}

/* ────────────────────────────────────────────────────────────────
 * Exposed RNG
 * ──────────────────────────────────────────────────────────────── */
void isildur_rng_init(isildur_rng_t *rng, uint64_t seed) {
  rng->state = seed ? seed : (uint64_t)time(NULL);
  if (!rng->state) rng->state = 1;
}
uint64_t isildur_rng_next(isildur_rng_t *rng) {
  rng->state = rng->state * 6364136223846793005ULL + 1;
  return rng->state;
}

/* ────────────────────────────────────────────────────────────────
 * Lifecycle
 * ──────────────────────────────────────────────────────────────── */
isildur_hv_t *isildur_alloc_hv(uint32_t dim) {
  isildur_hv_t *hv=calloc(1,sizeof(*hv));
  if(!hv)return NULL;
  hv->dim=dim; hv->n_words=ISILDUR_WORDS(dim);
  hv->bits=calloc(hv->n_words,sizeof(uint64_t));
  if(!hv->bits){free(hv);return NULL;}
  return hv;
}
void isildur_free_hv(isildur_hv_t *hv) { if(hv){free(hv->bits);free(hv);} }

isildur_hv_t *isildur_clone_hv(const isildur_hv_t *src) {
  isildur_hv_t *hv = isildur_alloc_hv(src->dim);
  if (!hv) return NULL;
  memcpy(hv->bits, src->bits, src->n_words * 8);
  return hv;
}

/* ────────────────────────────────────────────────────────────────
 * Generation
 * ──────────────────────────────────────────────────────────────── */
void isildur_gen_hv_rng(isildur_hv_t *hv, isildur_rng_t *rng) {
  if (rng) {
    for (uint32_t i = 0; i < hv->n_words; i++)
      hv->bits[i] = isildur_rng_next(rng) ^ (isildur_rng_next(rng) << 32);
  } else {
    for (uint32_t i = 0; i < hv->n_words; i++)
      hv->bits[i] = lcg_next() ^ (lcg_next() << 32);
  }
  uint32_t r = hv->dim % 64;
  if (r) hv->bits[hv->n_words - 1] &= ((1ULL << r) - 1);
}

void isildur_gen_hv(isildur_hv_t *hv, uint64_t seed) {
  lcg_seed(seed);
  for(uint32_t i=0;i<hv->n_words;i++)hv->bits[i]=lcg_next()^(lcg_next()<<32);
  uint32_t r=hv->dim%64; if(r)hv->bits[hv->n_words-1]&=((1ULL<<r)-1);
}

void isildur_gen_hv_batch(isildur_hv_t **hvs, uint32_t n, uint64_t base_seed) {
  isildur_rng_t rng;
  for (uint32_t i = 0; i < n; i++) {
    isildur_rng_init(&rng, base_seed + i);
    isildur_gen_hv_rng(hvs[i], &rng);
  }
}

void isildur_gen_balanced_hv(isildur_hv_t *hv, uint64_t seed) {
  isildur_gen_hv(hv,seed); isildur_balance(hv);
}

void isildur_balance(isildur_hv_t *hv) {
  uint32_t ones=isildur_popcount(hv), target=hv->dim/2;
  if(ones>target){uint32_t x=ones-target;
    for(uint32_t i=0;i<hv->dim&&x>0;i++)if(bit_get_internal(hv,i)){bit_set_internal(hv,i,0);x--;}}
  else if(ones<target){uint32_t d=target-ones;
    for(uint32_t i=0;i<hv->dim&&d>0;i++)if(!bit_get_internal(hv,i)){bit_set_internal(hv,i,1);d--;}}
}

/* ────────────────────────────────────────────────────────────────
 * Core HDC Operations
 * ──────────────────────────────────────────────────────────────── */
void isildur_bind(isildur_hv_t *r, const isildur_hv_t *a, const isildur_hv_t *b) {
  for(uint32_t i=0;i<a->n_words;i++)r->bits[i]=a->bits[i]^b->bits[i];
}
void isildur_unbind(isildur_hv_t *r, const isildur_hv_t *bnd, const isildur_hv_t *key) {
  isildur_bind(r,bnd,key);
}
void isildur_bundle(isildur_hv_t *r, const isildur_hv_t **hvs, uint32_t k) {
  if(!k){memset(r->bits,0,r->n_words*8);return;}
  uint32_t hk=k>>1;
  for(uint32_t i=0;i<r->dim;i++){uint32_t c=0;
    for(uint32_t v=0;v<k;v++)if(bit_get_internal(hvs[v],i))c++;
    bit_set_internal(r,i,c>hk);}
}

/* ────────────────────────────────────────────────────────────────
 * Incremental bundling with int32 accumulators
 * ──────────────────────────────────────────────────────────────── */
void isildur_bundle_int32_accumulate(int32_t *accums,
                                     const isildur_hv_t *hv,
                                     uint32_t dim, uint32_t *count) {
  for (uint32_t i = 0; i < dim; i++) {
    accums[i] += bit_get_internal(hv, i) ? 1 : -1;
  }
  (*count)++;
}

void isildur_bundle_int32_finalize(isildur_hv_t *result,
                                   const int32_t *accums,
                                   uint32_t dim, uint32_t count) {
  (void)count;
  for (uint32_t i = 0; i < dim; i++) {
    bit_set_internal(result, i, accums[i] >= 0 ? 1 : 0);
  }
}

/* ────────────────────────────────────────────────────────────────
 * Accumulate for per-class training (uses int32 per-bit sums)
 * ──────────────────────────────────────────────────────────────── */
void isildur_bundle_accumulate(const isildur_hv_t **accums, uint32_t *counts,
                               const isildur_hv_t *hv, uint32_t n_accums,
                               uint32_t class_idx) {
  if (class_idx >= n_accums) return;
  /* For each bit position, read the current running total,
   * add +1 or -1, and store back in the bits field.
   * We reuse bits as int32 accumulators (cast-safe for d <= 16384).
   * Strategy: use counts[class_idx] as weight and do majority.
   * For simplicity, we copy and re-bundle.
   */
  (void)accums; (void)counts; (void)hv; (void)n_accums; (void)class_idx;
  /* This is a more complex operation; use bundle_int32 family instead */
}

void isildur_bundle_finalize(isildur_hv_t *result,
                             const isildur_hv_t **accums,
                             const uint32_t *counts,
                             uint32_t class_idx) {
  uint32_t k = counts[class_idx];
  if (!k) { memset(result->bits, 0, result->n_words * 8); return; }
  uint32_t hk = k >> 1;
  for (uint32_t i = 0; i < result->dim; i++) {
    uint32_t c = bit_get_internal(accums[class_idx], i) ? counts[class_idx] : 0;
    /* accums not stored as int32 per bit; this is a stub for the HV-based accumulate */
    bit_set_internal(result, i, c > hk);
  }
}

/* ────────────────────────────────────────────────────────────────
 * Permute: cyclic right-shift
 * ──────────────────────────────────────────────────────────────── */
void isildur_permute(isildur_hv_t *r, const isildur_hv_t *hv, uint32_t sh) {
  sh%=hv->dim;
  for(uint32_t i=0;i<hv->dim;i++)bit_set_internal(r,i,bit_get_internal(hv,(i+sh)%hv->dim));
}

/* ────────────────────────────────────────────────────────────────
 * Similarity / Distance
 * ──────────────────────────────────────────────────────────────── */
uint32_t isildur_hamming(const isildur_hv_t *a, const isildur_hv_t *b) {
  uint32_t d=0; for(uint32_t i=0;i<a->n_words;i++)d+=pc64(a->bits[i]^b->bits[i]);
  return d;
}
uint32_t isildur_popcount(const isildur_hv_t *hv) {
  uint32_t c=0; for(uint32_t i=0;i<hv->n_words;i++)c+=pc64(hv->bits[i]); return c;
}
int32_t isildur_similarity(const isildur_hv_t *a, const isildur_hv_t *b) {
  uint32_t h=isildur_hamming(a,b); if(!a->dim)return 0;
  return (int32_t)(((int64_t)(a->dim-2*(int64_t)h)*1000)/(int64_t)a->dim);
}

void isildur_hamming_batch(const isildur_hv_t *query,
                           const isildur_hv_t **candidates,
                           uint32_t n_candidates,
                           uint32_t *distances) {
  for (uint32_t i = 0; i < n_candidates; i++) {
    distances[i] = isildur_hamming(query, candidates[i]);
  }
}

/* ────────────────────────────────────────────────────────────────
 * Associative Memory
 * ──────────────────────────────────────────────────────────────── */
uint32_t isildur_assoc_infer(const isildur_hv_t *q, const isildur_hv_t **chs,
                              uint32_t nc, uint32_t *dists) {
  uint32_t bc=0,bd=UINT32_MAX;
  for(uint32_t c=0;c<nc;c++){uint32_t d=isildur_hamming(q,chs[c]);
    if(dists)dists[c]=d; if(d<bd){bd=d;bc=c;}}
  return bc;
}
void isildur_assoc_topk(const isildur_hv_t *q, const isildur_hv_t **chs,
                         uint32_t nc, uint32_t k, uint32_t *ti, uint32_t *td) {
  for(uint32_t i=0;i<k;i++){ti[i]=UINT32_MAX;td[i]=UINT32_MAX;}
  for(uint32_t c=0;c<nc;c++){uint32_t d=isildur_hamming(q,chs[c]);
    for(uint32_t i=0;i<k;i++)if(d<td[i]){
      for(uint32_t j=k-1;j>i;j--){ti[j]=ti[j-1];td[j]=td[j-1];}
      ti[i]=c;td[i]=d;break;}}
}
void isildur_assoc_train(isildur_hv_t *chv, const isildur_hv_t *shv, uint32_t *cnt) {
  uint32_t c=*cnt+1,hc=c>>1;
  for(uint32_t i=0;i<chv->dim;i++){
    uint32_t e=bit_get_internal(chv,i)?*cnt:0,s=bit_get_internal(shv,i)?1:0;
    bit_set_internal(chv,i,e+s>hc);}
  *cnt=c;
}

/* ────────────────────────────────────────────────────────────────
 * Utility
 * ──────────────────────────────────────────────────────────────── */
void isildur_copy(isildur_hv_t *d, const isildur_hv_t *s) {
  d->dim=s->dim;d->n_words=s->n_words;memcpy(d->bits,s->bits,s->n_words*8);
}
bool isildur_equals(const isildur_hv_t *a, const isildur_hv_t *b) {
  if(a->dim!=b->dim)return false;
  for(uint32_t i=0;i<a->n_words;i++)if(a->bits[i]!=b->bits[i])return false;
  return true;
}
size_t isildur_serialize(const isildur_hv_t *hv, uint8_t *b, size_t sz) {
  size_t n=hv->n_words*8+4; if(sz<n)return 0;
  memcpy(b,&hv->dim,4);memcpy(b+4,hv->bits,hv->n_words*8); return n;
}
isildur_hv_t *isildur_deserialize(const uint8_t *b, size_t sz, uint32_t dim) {
  if(sz<4)return NULL;
  uint32_t sd; memcpy(&sd,b,4);
  if(dim&&sd!=dim)return NULL;
  isildur_hv_t *hv=isildur_alloc_hv(sd); if(!hv)return NULL;
  memcpy(hv->bits,b+4,hv->n_words*8); return hv;
}
void isildur_print(const isildur_hv_t *hv, uint32_t max) {
  uint32_t n=max&&max<hv->dim?max:hv->dim;
  printf("[");
  for(uint32_t i=0;i<n&&i<64;i++)printf("%c",bit_get_internal(hv,i)?'1':'0');
  if(n>64)printf("...");
  uint32_t pc=isildur_popcount(hv);
  printf("] dim=%u +1=%u -1=%u\n",hv->dim,pc,hv->dim-pc);
}

void isildur_bit_set(isildur_hv_t *hv, uint32_t i, uint64_t v) {
  bit_set_internal(hv, i, v);
}
uint64_t isildur_bit_get(const isildur_hv_t *hv, uint32_t i) {
  return bit_get_internal(hv, i);
}

/* ────────────────────────────────────────────────────────────────
 * Noise / Corruption
 * ──────────────────────────────────────────────────────────────── */
void isildur_corrupt_hv(isildur_hv_t *result,
                        const isildur_hv_t *src,
                        float rate,
                        isildur_noise_type_t type,
                        isildur_rng_t *rng) {
  isildur_rng_t local_rng;
  if (!rng) {
    isildur_rng_init(&local_rng, 0);
    rng = &local_rng;
  }

  /* Copy first */
  if (result != src) isildur_copy(result, src);

  uint32_t n_flip = (uint32_t)(rate * (float)result->dim);
  if (n_flip == 0 && rate > 0.0f) n_flip = 1;
  if (n_flip > result->dim) n_flip = result->dim;

  switch (type) {
    case ISILDUR_NOISE_FLIP:
    case ISILDUR_NOISE_SCALE:
      for (uint32_t i = 0; i < n_flip; i++) {
        uint32_t pos = (uint32_t)(isildur_rng_next(rng) % result->dim);
        uint64_t cur = bit_get_internal(result, pos);
        bit_set_internal(result, pos, cur ^ 1);
      }
      break;
    case ISILDUR_NOISE_DROP:
      for (uint32_t i = 0; i < n_flip; i++) {
        uint32_t pos = (uint32_t)(isildur_rng_next(rng) % result->dim);
        bit_set_internal(result, pos, 0);
      }
      break;
  }
}

/* ────────────────────────────────────────────────────────────────
 * Item Memory (scalar → HV quantization)
 * ──────────────────────────────────────────────────────────────── */
isildur_itemmem_t *isildur_itemmem_create(uint32_t n_levels, uint32_t dim,
                                           uint64_t seed, float flip_rate) {
  if (n_levels < 2 || dim == 0) return NULL;

  isildur_itemmem_t *im = calloc(1, sizeof(*im));
  if (!im) return NULL;
  im->n_levels = n_levels;
  im->dim = dim;

  im->level_hvs = calloc(n_levels, sizeof(isildur_hv_t *));
  if (!im->level_hvs) { free(im); return NULL; }

  /* Generate base level */
  im->level_hvs[0] = isildur_alloc_hv(dim);
  if (!im->level_hvs[0]) { isildur_itemmem_free(im); return NULL; }
  isildur_gen_balanced_hv(im->level_hvs[0], seed);

  isildur_rng_t rng;
  isildur_rng_init(&rng, seed + 42);

  for (uint32_t i = 1; i < n_levels; i++) {
    im->level_hvs[i] = isildur_clone_hv(im->level_hvs[i - 1]);
    if (!im->level_hvs[i]) { isildur_itemmem_free(im); return NULL; }

    /* Flip ~flip_rate fraction of bits from previous level */
    uint32_t n_flip = (uint32_t)(flip_rate * (float)dim);
    if (n_flip == 0 && flip_rate > 0.0f) n_flip = 1;
    /* Use bit flips at random positions to create gradual transition */
    for (uint32_t f = 0; f < n_flip; f++) {
      uint32_t pos = (uint32_t)(isildur_rng_next(&rng) % dim);
      uint64_t cur = bit_get_internal(im->level_hvs[i], pos);
      bit_set_internal(im->level_hvs[i], pos, cur ^ 1);
    }
    isildur_balance(im->level_hvs[i]);
  }

  return im;
}

void isildur_itemmem_free(isildur_itemmem_t *im) {
  if (!im) return;
  if (im->level_hvs) {
    for (uint32_t i = 0; i < im->n_levels; i++) {
      isildur_free_hv(im->level_hvs[i]);
    }
    free(im->level_hvs);
  }
  free(im);
}

isildur_hv_t *isildur_itemmem_encode_scalar(const isildur_itemmem_t *im,
                                              float value,
                                              float min_val, float max_val) {
  float denom = max_val - min_val;
  if (denom <= 0.0f) denom = 1.0f;
  float normalized = (value - min_val) / denom;
  if (normalized < 0.0f) normalized = 0.0f;
  if (normalized > 1.0f) normalized = 1.0f;

  int idx = (int)(normalized * (float)(im->n_levels - 1) + 0.5f);
  if (idx < 0) idx = 0;
  if (idx >= (int)im->n_levels) idx = (int)im->n_levels - 1;

  return isildur_clone_hv(im->level_hvs[idx]);
}

isildur_hv_t *isildur_itemmem_encode_vector(const isildur_itemmem_t *im,
                                              const float *values,
                                              uint32_t n_values,
                                              isildur_hv_t **key_hvs,
                                              float min_val, float max_val) {
  if (!n_values) return NULL;

  /* Find actual min/max if needed */
  if (min_val == 0.0f && max_val == 0.0f) {
    min_val = values[0];
    max_val = values[0];
    for (uint32_t i = 1; i < n_values; i++) {
      if (values[i] < min_val) min_val = values[i];
      if (values[i] > max_val) max_val = values[i];
    }
    if (max_val - min_val < 1e-6f) max_val = min_val + 1.0f;
  }

  /* Encode each scalar, bind with its key, and accumulate via int32 bundling */
  uint32_t dim = im->dim;
  int32_t *accums = calloc(dim, sizeof(int32_t));
  uint32_t count = 0;
  if (!accums) return NULL;

  for (uint32_t i = 0; i < n_values; i++) {
    isildur_hv_t *level_hv = isildur_itemmem_encode_scalar(
        im, values[i], min_val, max_val);
    if (!level_hv) { free(accums); return NULL; }

    /* Bind with key: bound = key_hvs[i] XOR level_hv */
    isildur_hv_t *bound = isildur_alloc_hv(dim);
    if (!bound) {
      isildur_free_hv(level_hv);
      free(accums);
      return NULL;
    }
    isildur_bind(bound, key_hvs[i], level_hv);
    isildur_free_hv(level_hv);

    isildur_bundle_int32_accumulate(accums, bound, dim, &count);
    isildur_free_hv(bound);
  }

  if (count == 0) { free(accums); return NULL; }

  isildur_hv_t *result = isildur_alloc_hv(dim);
  if (result) {
    isildur_bundle_int32_finalize(result, accums, dim, count);
  }
  free(accums);
  return result;
}

const isildur_hv_t *isildur_itemmem_get_level(const isildur_itemmem_t *im,
                                                uint32_t level) {
  if (!im || level >= im->n_levels) return NULL;
  return im->level_hvs[level];
}

/* ────────────────────────────────────────────────────────────────
 * CIM Hamming Distance
 * ──────────────────────────────────────────────────────────────── */
static isildur_cim_config_t cim_default_config(void) {
  isildur_cim_config_t c = {0};
  c.block_size = 32;
  c.n_classes = 0;
  c.hv_dim = 0;
  c.n_blocks = 0;
  c.process_variation = 0.0f;
  return c;
}

isildur_cim_t *isildur_cim_create(const isildur_cim_config_t *config) {
  isildur_cim_t *cim = calloc(1, sizeof(*cim));
  if (!cim) return NULL;

  if (config) {
    cim->config = *config;
  } else {
    cim->config = cim_default_config();
  }
  return cim;
}

void isildur_cim_free(isildur_cim_t *cim) {
  if (!cim) return;
  if (cim->class_hvs) {
    for (uint32_t i = 0; i < cim->config.n_classes; i++) {
      isildur_free_hv(cim->class_hvs[i]);
    }
    free(cim->class_hvs);
  }
  free(cim);
}

void isildur_cim_load(isildur_cim_t *cim,
                      isildur_hv_t **class_hvs,
                      uint32_t n_classes) {
  /* Free existing */
  if (cim->class_hvs) {
    for (uint32_t i = 0; i < cim->config.n_classes; i++)
      isildur_free_hv(cim->class_hvs[i]);
    free(cim->class_hvs);
  }

  cim->config.n_classes = n_classes;
  cim->config.hv_dim = n_classes > 0 ? class_hvs[0]->dim : 0;
  cim->config.n_blocks = cim->config.hv_dim > 0
    ? (cim->config.hv_dim + cim->config.block_size - 1) / cim->config.block_size
    : 0;

  cim->class_hvs = calloc(n_classes, sizeof(isildur_hv_t *));
  if (!cim->class_hvs) return;

  for (uint32_t i = 0; i < n_classes; i++) {
    cim->class_hvs[i] = isildur_clone_hv(class_hvs[i]);
  }
}

void isildur_cim_hamming(const isildur_cim_t *cim,
                         const isildur_hv_t *query,
                         uint32_t *distances) {
  uint32_t n_classes = cim->config.n_classes;

  /* Block-parallel CIM: for each block, compute partial Hamming across
   * all classes, then sum. This models the TCAM discharge-per-block model. */
  for (uint32_t c = 0; c < n_classes; c++) {
    distances[c] = 0;
  }

  uint32_t hv_dim = cim->config.hv_dim;
  uint32_t block_size = cim->config.block_size;
  uint32_t n_blocks = cim->config.n_blocks;
  uint32_t n_words_per_block = (block_size + 63) / 64;

  for (uint32_t b = 0; b < n_blocks; b++) {
    for (uint32_t c = 0; c < n_classes; c++) {
      for (uint32_t w = 0; w < n_words_per_block; w++) {
        uint32_t word_idx = b * n_words_per_block + w;
        if (word_idx >= query->n_words) break;
        uint64_t q_word = query->bits[word_idx];
        uint64_t c_word = cim->class_hvs[c]->bits[word_idx];

        /* Mask last word if needed */
        if (word_idx == query->n_words - 1) {
          uint32_t rem = hv_dim % 64;
          if (rem > 0) {
            uint64_t mask = (1ULL << rem) - 1;
            q_word &= mask;
            c_word &= mask;
          }
        }
        distances[c] += pc64(q_word ^ c_word);
      }
    }
  }

  /* Apply process variation if enabled */
  if (cim->config.process_variation > 0.0f) {
    isildur_rng_t rng;
    isildur_rng_init(&rng, 42);
    for (uint32_t c = 0; c < n_classes; c++) {
      if (((float)(isildur_rng_next(&rng) % 10000) / 10000.0f) < cim->config.process_variation) {
        /* Random ±1 error */
        if (isildur_rng_next(&rng) & 1 && distances[c] > 0) distances[c]--;
        else if (distances[c] < hv_dim) distances[c]++;
      }
    }
  }
}

uint32_t isildur_cim_predict(const isildur_cim_t *cim,
                              const isildur_hv_t *query,
                              uint32_t *dist) {
  uint32_t n_classes = cim->config.n_classes;
  uint32_t *dists = calloc(n_classes, sizeof(uint32_t));
  if (!dists) return 0;

  isildur_cim_hamming(cim, query, dists);

  uint32_t best_class = 0;
  uint32_t best_dist = UINT32_MAX;
  for (uint32_t c = 0; c < n_classes; c++) {
    if (dists[c] < best_dist) {
      best_dist = dists[c];
      best_class = c;
    }
  }
  if (dist) *dist = best_dist;
  free(dists);
  return best_class;
}

void isildur_cim_predict_batch(const isildur_cim_t *cim,
                               isildur_hv_t **queries,
                               uint32_t n,
                               uint32_t *predictions,
                               uint32_t *distances) {
  for (uint32_t i = 0; i < n; i++) {
    predictions[i] = isildur_cim_predict(cim, queries[i], distances ? &distances[i] : NULL);
  }
}

float isildur_cim_energy_pj(const isildur_cim_t *cim) {
  /* Based on Amrouch et al. 2022: precharge 20fJ/line, sense amp 10fJ/block */
  uint32_t n_lines = cim->config.n_classes * cim->config.n_blocks;
  float energy_precharge = 20.0f * (float)n_lines;  /* fJ */
  float energy_sense = 10.0f * (float)n_lines;       /* fJ */
  return (energy_precharge + energy_sense) / 1000.0f; /* pJ */
}

/* ────────────────────────────────────────────────────────────────
 * HD-Glue: Model Fusion
 * ──────────────────────────────────────────────────────────────── */
isildur_fusion_t *isildur_fusion_create(uint32_t n_models,
                                         uint32_t n_classes,
                                         uint32_t dim,
                                         uint64_t seed) {
  isildur_fusion_t *f = calloc(1, sizeof(*f));
  if (!f) return NULL;

  f->n_models = n_models;
  f->n_classes = n_classes;
  f->dim = dim;

  /* Allocate model IDs */
  f->model_ids = calloc(n_models, sizeof(isildur_hv_t *));
  if (!f->model_ids) { isildur_fusion_free(f); return NULL; }

  /* Allocate class HVs */
  f->class_hvs = calloc(n_classes, sizeof(isildur_hv_t *));
  if (!f->class_hvs) { isildur_fusion_free(f); return NULL; }

  /* Memory trace */
  f->memory_hv = isildur_alloc_hv(dim);
  if (!f->memory_hv) { isildur_fusion_free(f); return NULL; }

  /* Class accumulators */
  f->class_accum = calloc(n_classes, sizeof(isildur_hv_t *));
  f->class_counts = calloc(n_classes, sizeof(uint32_t));
  if (!f->class_accum || !f->class_counts) { isildur_fusion_free(f); return NULL; }

  /* Generate model_ids and class_hvs */
  if (n_models > 0) {
    isildur_hv_t **tmp_models = calloc(n_models, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_models; i++) {
      tmp_models[i] = isildur_alloc_hv(dim);
      isildur_gen_balanced_hv(tmp_models[i], seed + i * 1000);
    }
    for (uint32_t i = 0; i < n_models; i++) {
      f->model_ids[i] = tmp_models[i];
    }
    free(tmp_models);
  }

  if (n_classes > 0) {
    isildur_hv_t **tmp_classes = calloc(n_classes, sizeof(isildur_hv_t *));
    for (uint32_t i = 0; i < n_classes; i++) {
      tmp_classes[i] = isildur_alloc_hv(dim);
      isildur_gen_balanced_hv(tmp_classes[i], seed + 1 + i * 1000);
    }
    for (uint32_t i = 0; i < n_classes; i++) {
      f->class_hvs[i] = tmp_classes[i];
    }
    free(tmp_classes);
  }

  return f;
}

void isildur_fusion_free(isildur_fusion_t *f) {
  if (!f) return;
  if (f->model_ids) {
    for (uint32_t i = 0; i < f->n_models; i++) isildur_free_hv(f->model_ids[i]);
    free(f->model_ids);
  }
  if (f->class_hvs) {
    for (uint32_t i = 0; i < f->n_classes; i++) isildur_free_hv(f->class_hvs[i]);
    free(f->class_hvs);
  }
  isildur_free_hv(f->memory_hv);
  if (f->class_accum) {
    for (uint32_t i = 0; i < f->n_classes; i++) isildur_free_hv(f->class_accum[i]);
    free(f->class_accum);
  }
  free(f->class_counts);
  free(f);
}

void isildur_fusion_train(isildur_fusion_t *f,
                           uint32_t model_idx,
                           uint32_t class_idx,
                           float weight) {
  if (model_idx >= f->n_models || class_idx >= f->n_classes) return;
  if (weight <= 0.0f) return;

  /* Bind model ID with class HV, then add to memory trace */
  isildur_hv_t *bound = isildur_alloc_hv(f->dim);
  isildur_bind(bound, f->model_ids[model_idx], f->class_hvs[class_idx]);

  /* Weighted addition to memory: add weight * bound (per-bit with sign) */
  /* For binary, weight < 1.0 means probabilistic contribution */
  if (weight >= 1.0f) {
    /* Full contribution: int32 accumulate */
    int32_t *accums = calloc(f->dim, sizeof(int32_t));
    uint32_t dummy = 0;
    /* Convert memory_hv to int32 accum first, add bound, then finalize */
    for (uint32_t i = 0; i < f->dim; i++) {
      accums[i] = bit_get_internal(f->memory_hv, i) ? 1 : -1;
      accums[i] += bit_get_internal(bound, i) ? 1 : -1;
    }
    dummy = 2; /* not meaningful */
    isildur_bundle_int32_finalize(f->memory_hv, accums, f->dim, dummy);
    free(accums);
  } else {
    /* Probabilistic: use weight to decide contribution per bit */
    isildur_rng_t rng;
    isildur_rng_init(&rng, model_idx * 1000 + class_idx);
    for (uint32_t i = 0; i < f->dim; i++) {
      if (((float)(isildur_rng_next(&rng) % 10000) / 10000.0f) < weight) {
        uint64_t cur = bit_get_internal(f->memory_hv, i);
        uint64_t bv = bit_get_internal(bound, i);
        *((volatile int*)&cur); /* suppress unused */
        if (bv) {
          /* Add +1 influence: flip toward 1 */
          bit_set_internal(f->memory_hv, i, 1);
        } else {
          /* Add -1 influence: flip toward 0 */
          bit_set_internal(f->memory_hv, i, 0);
        }
      }
    }
  }

  /* Track per-class */
  if (!f->class_accum[class_idx]) f->class_accum[class_idx] = isildur_alloc_hv(f->dim);
  /* Accumulate to class_accum using int32 method */
  {
    int32_t *cacc = calloc(f->dim, sizeof(int32_t));
    for (uint32_t i = 0; i < f->dim; i++) {
      cacc[i] = bit_get_internal(f->class_accum[class_idx], i) ? 1 : -1;
      cacc[i] += bit_get_internal(bound, i) ? 1 : -1;
    }
    uint32_t dc = f->class_counts[class_idx] + 1;
    isildur_bundle_int32_finalize(f->class_accum[class_idx], cacc, f->dim, dc);
    free(cacc);
  }
  f->class_counts[class_idx]++;

  isildur_free_hv(bound);
}

/* Build a probe HV from model votes */
static isildur_hv_t *fusion_build_probe(const isildur_fusion_t *f,
                                         const uint32_t *model_votes,
                                         const float *probabilities,
                                         bool soft) {
  isildur_hv_t *probe = isildur_alloc_hv(f->dim);
  memset(probe->bits, 0, probe->n_words * 8);

  int32_t *accums = calloc(f->dim, sizeof(int32_t));
  uint32_t count = 0;

  for (uint32_t m = 0; m < f->n_models; m++) {
    for (uint32_t c = 0; c < f->n_classes; c++) {
      float w = 0.0f;
      if (soft) {
        w = probabilities[m * f->n_classes + c];
      } else {
        if (model_votes[m] == c) w = 1.0f;
      }
      if (w <= 0.0f) continue;

      isildur_hv_t *bound = isildur_alloc_hv(f->dim);
      isildur_bind(bound, f->model_ids[m], f->class_hvs[c]);

      for (uint32_t i = 0; i < f->dim; i++) {
        if (w >= 1.0f) {
          accums[i] += bit_get_internal(bound, i) ? 1 : -1;
        } else {
          accums[i] += (int32_t)(w * (float)(bit_get_internal(bound, i) ? 1 : -1));
        }
      }
      isildur_free_hv(bound);
      count++;
    }
  }

  if (count > 0) {
    isildur_bundle_int32_finalize(probe, accums, f->dim, count);
  }
  free(accums);
  return probe;
}

uint32_t isildur_fusion_predict_hard(const isildur_fusion_t *f,
                                      const uint32_t *model_votes,
                                      float *similarities) {
  isildur_hv_t *probe = fusion_build_probe(f, model_votes, NULL, false);

  /* Find class with highest similarity (lowest Hamming distance) */
  uint32_t best_class = 0;
  uint32_t best_dist = UINT32_MAX;
  for (uint32_t c = 0; c < f->n_classes; c++) {
    uint32_t d = isildur_hamming(probe, f->class_hvs[c]);
    if (similarities) {
      if (probe->dim > 0)
        similarities[c] = (float)(probe->dim - 2 * d) / (float)probe->dim;
      else
        similarities[c] = 0.0f;
    }
    if (d < best_dist) {
      best_dist = d;
      best_class = c;
    }
  }

  isildur_free_hv(probe);
  return best_class;
}

uint32_t isildur_fusion_predict_soft(const isildur_fusion_t *f,
                                      const float *probabilities,
                                      float *similarities) {
  isildur_hv_t *probe = fusion_build_probe(f, NULL, probabilities, true);

  uint32_t best_class = 0;
  uint32_t best_dist = UINT32_MAX;
  for (uint32_t c = 0; c < f->n_classes; c++) {
    uint32_t d = isildur_hamming(probe, f->class_hvs[c]);
    if (similarities) {
      if (probe->dim > 0)
        similarities[c] = (float)(probe->dim - 2 * d) / (float)probe->dim;
      else
        similarities[c] = 0.0f;
    }
    if (d < best_dist) {
      best_dist = d;
      best_class = c;
    }
  }

  isildur_free_hv(probe);
  return best_class;
}

void isildur_fusion_remove_model(isildur_fusion_t *f, uint32_t model_idx) {
  if (model_idx >= f->n_models) return;

  /* Subtract this model's bindings from memory */
  for (uint32_t c = 0; c < f->n_classes; c++) {
    isildur_hv_t *bound = isildur_alloc_hv(f->dim);
    isildur_bind(bound, f->model_ids[model_idx], f->class_hvs[c]);

    int32_t *accums = calloc(f->dim, sizeof(int32_t));
    for (uint32_t i = 0; i < f->dim; i++) {
      accums[i] = bit_get_internal(f->memory_hv, i) ? 1 : -1;
      accums[i] -= bit_get_internal(bound, i) ? 1 : -1;
    }
    uint32_t dummy = 1;
    isildur_bundle_int32_finalize(f->memory_hv, accums, f->dim, dummy);
    free(accums);
    isildur_free_hv(bound);
  }
}

float isildur_fusion_model_contribution(const isildur_fusion_t *f,
                                         uint32_t model_idx) {
  if (model_idx >= f->n_models) return 0.0f;

  /* Build model's full contribution */
  isildur_hv_t *contrib = isildur_alloc_hv(f->dim);
  memset(contrib->bits, 0, contrib->n_words * 8);

  int32_t *accums = calloc(f->dim, sizeof(int32_t));
  for (uint32_t c = 0; c < f->n_classes; c++) {
    isildur_hv_t *bound = isildur_alloc_hv(f->dim);
    isildur_bind(bound, f->model_ids[model_idx], f->class_hvs[c]);
    for (uint32_t i = 0; i < f->dim; i++) {
      accums[i] += bit_get_internal(bound, i) ? 1 : -1;
    }
    isildur_free_hv(bound);
  }
  isildur_bundle_int32_finalize(contrib, accums, f->dim, f->n_classes);
  free(accums);

  /* Cosine similarity between contrib and memory */
  uint32_t h = isildur_hamming(contrib, f->memory_hv);
  float sim = 0.0f;
  if (f->dim > 0) sim = (float)(f->dim - 2 * h) / (float)f->dim;

  isildur_free_hv(contrib);
  return sim;
}

void isildur_fusion_normalize(isildur_fusion_t *f) {
  /* Threshold memory to bipolar ∈ {-1, +1} */
  for (uint32_t i = 0; i < f->dim; i++) {
    /* For memory_hv, bits are already bipolar (1=+1, 0=-1).
     * We threshold by majority: if majority of accumulated contributions
     * are positive, bit=1. But since memory_hv is already thresholded
     * during accumulation, this is a no-op for bipolar mode.
     * We just ensure balance. */
    (void)i; /* Already bipolar by construction from bundle_int32_finalize */
  }
  isildur_balance(f->memory_hv);
}

/* ────────────────────────────────────────────────────────────────
 * HDC Encoder (SpikeHDC + Associative Memory)
 * ──────────────────────────────────────────────────────────────── */
isildur_encoder_t *isildur_encoder_create(uint32_t input_size,
                                            uint32_t n_classes,
                                            uint32_t dim,
                                            uint32_t n_levels,
                                            uint64_t seed) {
  if (input_size == 0 || dim == 0) return NULL;

  isildur_encoder_t *enc = calloc(1, sizeof(*enc));
  if (!enc) return NULL;

  enc->input_size = input_size;
  enc->n_classes = n_classes;
  enc->dim = dim;

  /* Create item memory */
  enc->item_mem = isildur_itemmem_create(n_levels, dim, seed, 0.05f);
  if (!enc->item_mem) { isildur_encoder_free(enc); return NULL; }

  /* Create key hypervectors (one per input feature) */
  enc->keys = calloc(input_size, sizeof(isildur_hv_t *));
  if (!enc->keys) { isildur_encoder_free(enc); return NULL; }
  for (uint32_t i = 0; i < input_size; i++) {
    enc->keys[i] = isildur_alloc_hv(dim);
    if (!enc->keys[i]) { isildur_encoder_free(enc); return NULL; }
    isildur_gen_balanced_hv(enc->keys[i], seed + 1000 + i);
  }

  /* Create class hypervectors */
  enc->class_hvs = calloc(n_classes, sizeof(isildur_hv_t *));
  enc->class_counts = calloc(n_classes, sizeof(uint32_t));
  if (!enc->class_hvs || !enc->class_counts) { isildur_encoder_free(enc); return NULL; }
  for (uint32_t i = 0; i < n_classes; i++) {
    enc->class_hvs[i] = isildur_alloc_hv(dim);
    if (!enc->class_hvs[i]) { isildur_encoder_free(enc); return NULL; }
    memset(enc->class_hvs[i]->bits, 0, enc->class_hvs[i]->n_words * 8);
  }

  return enc;
}

void isildur_encoder_free(isildur_encoder_t *enc) {
  if (!enc) return;
  isildur_itemmem_free(enc->item_mem);
  if (enc->keys) {
    for (uint32_t i = 0; i < enc->input_size; i++) isildur_free_hv(enc->keys[i]);
    free(enc->keys);
  }
  if (enc->class_hvs) {
    for (uint32_t i = 0; i < enc->n_classes; i++) isildur_free_hv(enc->class_hvs[i]);
    free(enc->class_hvs);
  }
  free(enc->class_counts);
  free(enc);
}

isildur_hv_t *isildur_encoder_encode(const isildur_encoder_t *enc,
                                       const float *inputs) {
  /* Find min/max of inputs for range normalization */
  float min_val = inputs[0], max_val = inputs[0];
  for (uint32_t i = 1; i < enc->input_size; i++) {
    if (inputs[i] < min_val) min_val = inputs[i];
    if (inputs[i] > max_val) max_val = inputs[i];
  }
  if (max_val - min_val < 1e-6f) max_val = min_val + 1.0f;

  return isildur_itemmem_encode_vector(
      enc->item_mem, inputs, enc->input_size, enc->keys, min_val, max_val);
}

void isildur_encoder_train(isildur_encoder_t *enc,
                            const float *inputs,
                            uint32_t class_idx) {
  if (class_idx >= enc->n_classes) return;

  isildur_hv_t *sample_hv = isildur_encoder_encode(enc, inputs);
  if (!sample_hv) return;

  /* Accumulate using int32 per-bit method */
  int32_t *accums = calloc(enc->dim, sizeof(int32_t));
  for (uint32_t i = 0; i < enc->dim; i++) {
    /* Reconstruct current accums from class_hvs and counts */
    if (enc->class_counts[class_idx] > 0) {
      accums[i] = bit_get_internal(enc->class_hvs[class_idx], i)
                    ? (int32_t)enc->class_counts[class_idx]
                    : -(int32_t)enc->class_counts[class_idx];
    }
  }
  isildur_bundle_int32_accumulate(accums, sample_hv, enc->dim, &enc->class_counts[class_idx]);
  isildur_bundle_int32_finalize(enc->class_hvs[class_idx], accums, enc->dim,
                                 enc->class_counts[class_idx]);
  free(accums);
  isildur_free_hv(sample_hv);
}

uint32_t isildur_encoder_predict(const isildur_encoder_t *enc,
                                   const float *inputs,
                                   uint32_t *dist) {
  isildur_hv_t *query = isildur_encoder_encode(enc, inputs);
  if (!query) return 0;

  uint32_t best_class = 0;
  uint32_t best_dist = UINT32_MAX;
  for (uint32_t c = 0; c < enc->n_classes; c++) {
    uint32_t d = isildur_hamming(query, enc->class_hvs[c]);
    if (d < best_dist) {
      best_dist = d;
      best_class = c;
    }
  }
  if (dist) *dist = best_dist;
  isildur_free_hv(query);
  return best_class;
}

void isildur_encoder_finalize(isildur_encoder_t *enc) {
  /* Threshold class HVs to bipolar */
  for (uint32_t c = 0; c < enc->n_classes; c++) {
    if (enc->class_counts[c] > 0) {
      isildur_balance(enc->class_hvs[c]);
    } else {
      /* Unseen class: generate random */
      isildur_gen_balanced_hv(enc->class_hvs[c], 9999 + c);
    }
  }
}

uint32_t isildur_encoder_predict_scores(const isildur_encoder_t *enc,
                                          const float *inputs,
                                          int32_t *similarities) {
  isildur_hv_t *query = isildur_encoder_encode(enc, inputs);
  if (!query) return 0;

  uint32_t best_class = 0;
  int32_t best_sim = INT32_MIN; /* highest cosine similarity */
  for (uint32_t c = 0; c < enc->n_classes; c++) {
    int32_t sim = isildur_similarity(query, enc->class_hvs[c]);
    if (similarities) similarities[c] = sim;
    if (sim > best_sim) {
      best_sim = sim;
      best_class = c;
    }
  }
  isildur_free_hv(query);
  return best_class;
}