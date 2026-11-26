// Copyright 2026 DeepBeepMeep. SPDX-License-Identifier: Apache-2.0
// A wave owns sixteen queries. Skip empty key tiles before loading either matrix.
// The same online softmax includes the exact keys and the weighted representatives.
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
namespace mat = rocwmma;
using Half = rocwmma::float16_t;
#else
#include <cuda_fp16.h>
#include <mma.h>
namespace mat = nvcuda::wmma;
using Half = half;
#endif
#include <stdint.h>

extern "C" __global__ void sg_pack(const Half* x, int8_t* out, float* scale, int width) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float peak = 0.f;
    for (int c = lane; c < width; c += warpSize)
        peak = fmaxf(peak, fabsf(float(x[(int64_t)row * width + c])));
    for (int d = warpSize/2; d; d /= 2) {
#ifdef __HIPCC__
        peak = fmaxf(peak, __shfl_xor(peak, d, warpSize));
#else
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, d));
#endif
    }
    const float step = peak > 0.f ? peak / 127.f : 1.f;
    if (!lane) scale[row] = step;
    for (int c = lane; c < width; c += warpSize)
        out[(int64_t)row * width + c] = (int8_t)rintf(float(x[(int64_t)row * width + c]) / step);
}

template<int D>
__device__ void attend(const int8_t* q, const int8_t* k, const Half* v,
                       const float* qs, const float* ks, const uint32_t* bits, Half* output,
                       int qpad, int kpad, int keys, int qb, float scale, int reps, int causal) {
    const int group = blockIdx.y, first = blockIdx.x * 16, lane = threadIdx.x;
    const int nk = (keys + 63) / 64, nq = (qpad + qb - 1) / qb;
    const int exact = nk * 64;
    q += ((int64_t)group * qpad + first) * D;
    qs += (int64_t)group * qpad + first;
    k += (int64_t)group * kpad * D;
    v += (int64_t)group * kpad * D;
    ks += (int64_t)group * kpad;
    bits += ((int64_t)group * nq + first / qb) * nk * 2;
    output += ((int64_t)group * qpad + first) * D;

    __shared__ __align__(32) int product[16*64];
    __shared__ __align__(32) Half prob[16*64];
    __shared__ __align__(32) float tile[16*16], numerator[16*D], maximum[16], denom[16], alpha[16];
    __shared__ float bias[64], scores[16*64];
    for (int i = lane; i < 16 * D; i += warpSize) numerator[i] = 0.f;
    if (lane < 16) { maximum[lane] = -INFINITY; denom[lane] = 0.f; }
    __syncthreads();
    mat::fragment<mat::matrix_a,16,16,16,int8_t,mat::row_major> query[D/16];
    for (int d = 0; d < D; d += 16) mat::load_matrix_sync(query[d/16], q+d, D);
    for (int start = 0; start < (reps ? exact+nk : keys); start += 64) {
        const bool representative = start >= exact;
        uint32_t lo = 0xffffffffu, hi = 0xffffffffu;
        if (!representative) {
            lo = bits[start/32]; hi = bits[start/32+1];
            if (!(lo | hi)) continue;
        }
        for (int c=lane;c<64;c+=warpSize) {
            int col=start+c, mass;
            if (representative) {
                int j=col-exact;
                mass=j<nk ? min(64,keys-j*64)-__popc(bits[2*j])-__popc(bits[2*j+1]) : 0;
            } else mass=col<keys && ((c<32 ? lo : hi) & (1u<<(c%32)));
            bias[c]=mass>0 ? (representative ? logf(float(mass)) : 0.f) : -INFINITY;
        }
        for (int c=0;c<64;c+=16) {
            mat::fragment<mat::accumulator,16,16,16,int> score;
            mat::fill_fragment(score,0);
            if (((c<32 ? lo : hi)>>(c%32)) & 0xffffu) {
                for (int d=0; d<D; d+=16) {
                    mat::fragment<mat::matrix_b,16,16,16,int8_t,mat::col_major> key;
                    mat::load_matrix_sync(key,k+(int64_t)(start+c)*D+d,D);
                    mat::mma_sync(score,query[d/16],key,score);
                }
            }
            mat::store_matrix_sync(product+c,score,64,mat::mem_row_major);
        }
        __syncthreads();
        for (int i=lane;i<16*64;i+=warpSize) {
            int r=i/64,c=i%64;
            scores[i]=causal && start+c>first+r ? -INFINITY : product[i]*(qs[r]*ks[start+c]*scale)+bias[c];
        }
        __syncthreads();
        if (lane<16) {
            float m=maximum[lane];
            for (int c=0;c<64;c++) m=fmaxf(m,scores[lane*64+c]);
            const float safe=isfinite(m) ? m : 0.f;
            const float a=expf(maximum[lane]-safe);
            float sum=denom[lane]*a;
            for (int c=0;c<64;c++) {
                float p=expf(scores[lane*64+c]-safe);
                prob[lane*64+c]=Half(p); sum+=p;
            }
            maximum[lane]=m; denom[lane]=sum; alpha[lane]=a;
        }
        __syncthreads();
        mat::fragment<mat::matrix_a,16,16,16,Half,mat::row_major> probability[4];
        for (int c=0;c<64;c+=16) mat::load_matrix_sync(probability[c/16],prob+c,64);
        for (int d=0;d<D;d+=16) {
            mat::fragment<mat::accumulator,16,16,16,float> pv;
            mat::fill_fragment(pv,0.f);
            for (int c=0;c<64;c+=16) {
                if (((c<32 ? lo : hi)>>(c%32)) & 0xffffu) {
                    mat::fragment<mat::matrix_b,16,16,16,Half,mat::row_major> value;
                    mat::load_matrix_sync(value,v+(int64_t)(start+c)*D+d,D);
                    mat::mma_sync(pv,probability[c/16],value,pv);
                }
            }
            mat::store_matrix_sync(tile,pv,16,mat::mem_row_major);
            __syncthreads();
            for (int i=lane;i<256;i+=warpSize) {
                const int r=i/16, ix=r*D+d+i%16;
                numerator[ix]=fmaf(numerator[ix],alpha[r],tile[i]);
            }
            __syncthreads();
        }
    }
    for (int i=lane; i<16*D; i+=warpSize)
        output[i] = Half(denom[i/D] > 0.f ? numerator[i]/denom[i/D] : 0.f);
}
#define ENTRY(D) extern "C" __global__ void sg_attn##D(const int8_t* q,const int8_t* k,const Half* v,const float* qs,const float* ks,const uint32_t* bits,Half* o,int qp,int kp,int keys,int qb,float scale,int reps,int causal) { attend<D>(q,k,v,qs,ks,bits,o,qp,kp,keys,qb,scale,reps,causal); }
ENTRY(64)
ENTRY(128)

// Score sixteen block means against one 64-token key block. Only mixed
// tiles read K. The second pass packs 64 exact-token decisions in two words.
extern "C" __global__ void sg_select(const Half* qm, const Half* k, const int8_t* flags,
                                     float* moments, const float* cutoff, uint32_t* bits,
                                     int nq, int padded_q, int tk, int dim, float scale, int select) {
    const int g=blockIdx.z, kb=blockIdx.y, baseq=blockIdx.x*16, lane=threadIdx.x;
    const int nk=(tk+63)/64, len=min(64,tk-kb*64);
    const int64_t row=(int64_t)g*nq+baseq;
    __shared__ int choices[16];
    __shared__ __align__(32) float scores[16*64];
    if (lane<16) choices[lane]=baseq+lane<nq ? flags[(row+lane)*nk+kb] : 0;
    __syncthreads();
    bool mixed=false;
    for (int r=0;r<16;r++) mixed |= choices[r]==1;
    if (mixed) {
        for (int c=0;c<64;c+=16) {
            mat::fragment<mat::accumulator,16,16,16,float> dot;
            mat::fill_fragment(dot,0.f);
            for (int d=0;d<dim;d+=16) {
                mat::fragment<mat::matrix_a,16,16,16,Half,mat::row_major> a;
                mat::fragment<mat::matrix_b,16,16,16,Half,mat::col_major> b;
                mat::load_matrix_sync(a,qm+((int64_t)g*padded_q+baseq)*dim+d,dim);
                mat::load_matrix_sync(b,k+(((int64_t)g*nk+kb)*64+c)*dim+d,dim);
                mat::mma_sync(dot,a,b,dot);
            }
            mat::store_matrix_sync(scores+c,dot,64,mat::mem_row_major);
        }
    }
    __syncthreads();
    if (!select && lane<16 && baseq+lane<nq) {
        float sum=0.f, sum2=0.f;
        const int count=choices[lane]==1 ? len : 0;
        for (int c=0;c<count;c++) {
            float x=scores[lane*64+c]*scale;
            sum+=x; sum2=fmaf(x,x,sum2);
        }
        const int64_t offset=((row+lane)*nk+kb)*3;
        moments[offset]=float(count); moments[offset+1]=sum; moments[offset+2]=sum2;
    } else if (select && lane<32 && baseq+lane/2<nq) {
        const int r=lane/2, first=(lane%2)*32, flag=choices[r];
        uint32_t mask=0;
        if (flag!=0) {
            const float threshold=cutoff[row+r];
            for (int c=first;c<min(first+32,len);c++)
                if (flag<0 || scores[r*64+c]*scale>threshold) mask |= 1u<<(c-first);
        }
        bits[((row+r)*nk+kb)*2+lane%2]=mask;
    }
}
