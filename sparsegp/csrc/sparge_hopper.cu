// Stable driver entry points for upstream 64 x 128 WGMMA SpargeAttention.
#include <type_traits>
#define SPARSEGP_DEVICE_ABI
#include "../../csrc/qattn/qk_int_sv_f8_cuda_sm90.cuh"

#define ENTRY(NAME,D,T,CAUSAL) \
extern "C" __global__ void NAME(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, \
        const __grid_constant__ CUtensorMap v, float* qs, float* ks, float* vs, T* out, int* lut, \
        int* count, float* pv, unsigned ob, unsigned oh, unsigned ot, unsigned nq, unsigned nk, float scale) { \
    constexpr auto gran=QuantGranularity::kPerBlock; \
    qk_int8_sv_f8_attn_kernel<64,128,128,D,gran,gran,PVThresholdMode::kPerBlock,T, \
        CAUSAL ? MaskMode::kCausal : MaskMode::kNone,true,false>(q,k,v,qs,ks,vs,out,nullptr,lut,count,pv,ob,oh,ot,nq,nk,1,scale); \
    const unsigned group=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x; \
    const int first=count[group] ? (CAUSAL ? lut[group*((nk+127)/128)]*128 : 0) : nq; \
    int rows=min(64,first-int(blockIdx.x*64)); \
    if (rows>0) { \
        __syncthreads(); \
        for (int i=threadIdx.x;i<rows*D/8;i+=128) { \
            unsigned row=blockIdx.x*64+i/(D/8); \
            if (row<nq) ((uint4*)(out+blockIdx.z*ob+blockIdx.y*oh+row*ot))[i%(D/8)]=make_uint4(0,0,0,0); \
        } \
    } \
}
#define BOTH(D,T,SUFFIX) ENTRY(sparge_##D##_##SUFFIX,D,T,false) ENTRY(sparge_##D##_##SUFFIX##_causal,D,T,true)
BOTH(64,half,f16)
BOTH(128,half,f16)
BOTH(64,nv_bfloat16,bf16)
BOTH(128,nv_bfloat16,bf16)
