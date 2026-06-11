/**
 * isildur.c — Isildur HDC/VSA Core Library (C Reference Implementation)
 * Portable C99. No dependencies beyond libc.
 */
#include "isildur.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

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
static inline uint32_t pc64(uint64_t x) {
  return (uint32_t)(pop8[x&0xFF]+pop8[(x>>8)&0xFF]+pop8[(x>>16)&0xFF]+
    pop8[(x>>24)&0xFF]+pop8[(x>>32)&0xFF]+pop8[(x>>40)&0xFF]+
    pop8[(x>>48)&0xFF]+pop8[(x>>56)&0xFF]);
}

static inline uint64_t bit_get(const isildur_hv_t *hv, uint32_t i) {
  return (hv->bits[i/64] >> (i%64)) & 1ULL;
}
static inline void bit_set(isildur_hv_t *hv, uint32_t i, uint64_t v) {
  uint32_t w=i/64,s=i%64; hv->bits[w]=(hv->bits[w]&~(1ULL<<s))|(v<<s);
}

static uint64_t lcg_s=1;
static inline void lcg_seed(uint64_t s) { lcg_s=s?s:(uint64_t)time(NULL); if(!lcg_s)lcg_s=1; }
static inline uint64_t lcg_next(void) { lcg_s=lcg_s*6364136223846793005ULL+1; return lcg_s; }

isildur_hv_t *isildur_alloc_hv(uint32_t dim) {
  isildur_hv_t *hv=calloc(1,sizeof(*hv));
  if(!hv)return NULL;
  hv->dim=dim; hv->n_words=ISILDUR_WORDS(dim);
  hv->bits=calloc(hv->n_words,sizeof(uint64_t));
  if(!hv->bits){free(hv);return NULL;}
  return hv;
}
void isildur_free_hv(isildur_hv_t *hv) { if(hv){free(hv->bits);free(hv);} }

void isildur_gen_hv(isildur_hv_t *hv, uint64_t seed) {
  lcg_seed(seed);
  for(uint32_t i=0;i<hv->n_words;i++)hv->bits[i]=lcg_next()^(lcg_next()<<32);
  uint32_t r=hv->dim%64; if(r)hv->bits[hv->n_words-1]&=((1ULL<<r)-1);
}
void isildur_gen_balanced_hv(isildur_hv_t *hv, uint64_t seed) {
  isildur_gen_hv(hv,seed); isildur_balance(hv);
}
void isildur_balance(isildur_hv_t *hv) {
  uint32_t ones=isildur_popcount(hv), target=hv->dim/2;
  if(ones>target){uint32_t x=ones-target;
    for(uint32_t i=0;i<hv->dim&&x>0;i++)if(bit_get(hv,i)){bit_set(hv,i,0);x--;}}
  else if(ones<target){uint32_t d=target-ones;
    for(uint32_t i=0;i<hv->dim&&d>0;i++)if(!bit_get(hv,i)){bit_set(hv,i,1);d--;}}
}

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
    for(uint32_t v=0;v<k;v++)if(bit_get(hvs[v],i))c++;
    bit_set(r,i,c>hk);}
}
void isildur_permute(isildur_hv_t *r, const isildur_hv_t *hv, uint32_t sh) {
  sh%=hv->dim;
  for(uint32_t i=0;i<hv->dim;i++)bit_set(r,i,bit_get(hv,(i+sh)%hv->dim));
}

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
    uint32_t e=bit_get(chv,i)?*cnt:0,s=bit_get(shv,i)?1:0;
    bit_set(chv,i,e+s>hc);}
  *cnt=c;
}

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
  for(uint32_t i=0;i<n&&i<64;i++)printf("%c",bit_get(hv,i)?'1':'0');
  if(n>64)printf("...");
  uint32_t pc=isildur_popcount(hv);
  printf("] dim=%u +1=%u -1=%u\n",hv->dim,pc,hv->dim-pc);
}
