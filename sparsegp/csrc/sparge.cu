// Device-only entry points for the upstream Apache-2.0 SpargeAttention tensor-core kernels.
// The output is strided NHD (as in SageGP) so a head group can write directly into a projection's input.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <type_traits>
#define SPARSEGP_DEVICE_ABI
#if SPARGE_FP8
#include "../../csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh"
#else
#include "../../csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh"
#endif

#define PARAMETERS(TYPE) int8_t* q, int8_t* k, void* v, TYPE* out, int* lut, int* count, float* pv, \
    float* qs, float* ks, float* vs, unsigned nq, unsigned nk, unsigned v0, unsigned v1, unsigned v2, \
    unsigned o0, unsigned o1, unsigned o2, float scale

template<int D, class T, bool Causal, int Rows=128>
__device__ __forceinline__ void sparse_tile(PARAMETERS(T)) {
    const unsigned h = gridDim.y;
    const unsigned row_group = (blockIdx.z*h + blockIdx.y)*((nq+127)/128) + blockIdx.x*Rows/128;
    const unsigned keys = (nk + 63)/64;
    const int used = count[row_group];
    int first_visible = 0;
    if (!used || Causal) {
        first_visible = used ? lut[row_group*keys]*64 : nq;
    }
    if (used) {
        constexpr auto mask = Causal ? MaskMode::kCausal : MaskMode::kNone;
        constexpr auto gran = QuantGranularity::kPerBlock;
        constexpr auto prune = PVThresholdMode::kPerBlock;
#if SPARGE_FP8
        qk_int_sv_f8_block_sparse_attn_kernel<Rows,64,(Rows==64?16:32),64,D,DataType::kInt8,gran,gran,
            float,true,SPARGE_HALF_PV,prune,T,ComputeUnit::kCudaCore,mask,true,false,128>
            (q,k,(int8_t*)v,out,nullptr,lut,count,pv,qs,ks,vs,nq,nk,1,
             nq*h*D,h*D,D,nk*h*D,h*D,D,v0,v1,v2,o0,o1,o2,scale);
#else
        qk_int_sv_f16_block_sparse_attn_kernel<128,64,(D==64?32:16),64,D,DataType::kInt8,gran,gran,
            true,prune,T,ComputeUnit::kTensorCore,mask,false>
            (q,k,(half*)v,out,nullptr,lut,count,pv,qs,ks,nq,nk,1,
             nq*h*D,h*D,D,nk*h*D,h*D,D,v0,v1,v2,o0,o1,o2,scale);
#endif
    }
    // Write empty rows in the same launch, including early causal queries with no visible keys.
    int empty = min(Rows, first_visible - int(blockIdx.x*Rows));
    if (empty > 0) {
        if (used) __syncthreads();
        int lane = threadIdx.y*32 + threadIdx.x;
        for (int i=lane; i<empty*D/8; i+=blockDim.y*32) {
            unsigned row = blockIdx.x*Rows + i/(D/8);
            if (row<nq) ((uint4*)(out+blockIdx.z*o0+row*o1+blockIdx.y*o2))[i%(D/8)]=make_uint4(0,0,0,0);
        }
    }
}
#define VARIANT(NAME,D,T,C,ROWS) extern "C" __global__ void NAME(PARAMETERS(T)) { \
    sparse_tile<D,T,C,ROWS>(q,k,v,out,lut,count,pv,qs,ks,vs,nq,nk,v0,v1,v2,o0,o1,o2,scale); }
#define ENTRY(NAME,D,T,C) VARIANT(NAME,D,T,C,128)
#define BOTH(D,T,SUFFIX) ENTRY(sparge_##D##_##SUFFIX,D,T,false) ENTRY(sparge_##D##_##SUFFIX##_causal,D,T,true)
BOTH(64,half,f16)
BOTH(128,half,f16)
BOTH(64,nv_bfloat16,bf16)
BOTH(128,nv_bfloat16,bf16)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
// Two CTAs share a 128-row routing decision. The smaller 64-row CTA fills short, head-split launches.
VARIANT(sparge_64_f16_q64,64,half,false,64)
VARIANT(sparge_64_bf16_q64,64,nv_bfloat16,false,64)
VARIANT(sparge_128_f16_q64,128,half,false,64)
VARIANT(sparge_128_bf16_q64,128,nv_bfloat16,false,64)
#endif
