# SparseGP

**Sparse attention for the GPU Poor.** SparseGP brings Sol³ mixed-block routing with INT8 attention and SpargeAttention block masks to [WanGP](https://github.com/deepbeepmeep/Wan2GP). It ships GPU machine code for GeForce RTX 20–50 series and Radeon RDNA 3/4. The same wheel runs on Windows and Linux.

The fork is based on [SpargeAttention](https://github.com/thu-ml/SpargeAttn). Its original documentation, paper and citation are preserved in [README_SpargeAttention.md](README_SpargeAttention.md). The hierarchical routing method comes from [Sol³ by Fan and coauthors](https://github.com/FofGofx/sol3). WanGP contributed the packed INT8 and NVFP4 paths.

## Attention modes

| Mode | Inputs | Behavior |
|---|---|---|
| `sol3_attention(q, k, v)` | NHD, bidirectional self-attention, FP16 or BF16, head widths 64 and 128, at least 1,024 tokens | INT8 scores. Important blocks keep all keys. Mixed blocks keep selected keys. Each remaining block contributes its mean key and value with the mass of its remaining tokens. |
| `sol3_attention(..., precision="fp4")` | SM120, eligible BF16 width-128 inputs, CUDA 13 class driver with PyTorch's image loader | NVFP4 scores and FP8 values with the same routing. INT8 is the default. |
| `sparge_attention(q, k, v, block_map)` | HND, boolean or 0/1 block map | Attend to the keys allowed by the map. Supports causal masking, rectangular attention and partial final blocks. An empty row returns zero. |
| `spas_sage2_attn_meansim_topk_cuda(...)` | Original SpargeAttention layout and selection arguments | Builds the block map from pooled scores and within-block similarity. The CDF variant is also available. |

For Sol³, a protected prefix or suffix receives dense attention and its key blocks remain exact. Short sequences and incompatible shapes use dense attention. A dense Sparge call uses [SageGP](https://github.com/deepbeepmeep/SageGP) when its kernel supports the call. PyTorch SDPA handles other dense requests, including explicit token masks. With an explicit block map SparseGP preserves the map even if the fast operator cannot serve the shape.

### Measured

On the Wan 2.2 I2V singer clip, SparseGP Sol3 INT8 took **13.99 seconds** over four transformer steps. Sage2 took **15.10 seconds**, upstream Sol³ BF16 took **14.05 seconds**, and the existing Sol INT8 path took **12.56 seconds**. All four used the same seed, conditioning image, model, sampler and lossless output.

| Wan 2.2 I2V, 832 × 480, 81 frames | Stone ring LPIPS | Singer LPIPS | Singer transformer time |
|---|---:|---:|---:|
| Sage2 reference | 0 | 0 | 15.10 s |
| Sol INT8 | 0.0159 | 0.0570 | 12.56 s |
| Sol FP4 | 0.0182 | 0.0607 | 12.38 s |
| Upstream Sol³ BF16 | 0.0135 | 0.0639 | 14.05 s |
| SparseGP Sol3 INT8 | 0.0105 | 0.0422 | 13.99 s |
| SparseGP Sol3 FP4 | 0.0149 | 0.0494 | 13.60 s |

| MiniMax H3, 1280 × 704, 124 frames | Clockmaker LPIPS | Fox LPIPS | Fox transformer time |
|---|---:|---:|---:|
| Sage2 reference | 0.0000 | 0.0000 | 31.32 s |
| Sol INT8 | 0.3867 | 0.2847 | 24.10 s |
| Sol FP4 | 0.3816 | 0.2901 | 23.58 s |
| Upstream Sol³ BF16 | 0.2756 | 0.2715 | 30.88 s |
| SparseGP Sol3 INT8 | 0.2776 | 0.2444 | 29.26 s |
| SparseGP Sol3 FP4 | 0.3057 | 0.3014 | 28.10 s |

H3 used six denoising steps with the same dense decoder for every method. The larger perceptual distances on H3 show how strongly the model and scene affect the result.

LPIPS measures distance from the Sage2 clip. Lower values indicate closer agreement. The measurements use every sixth decoded frame at 448 pixels wide. Timings sum synchronized transformer calls from one generation per cell on SM120 and exclude text encoding, VAE decoding and file output. The complete [per-scene timing and quality results](https://github.com/deepbeepmeep/SparseGP/releases/download/v1.0.0/benchmarks.json) include SSIM, RGB PSNR and individual step times.

## Installation

Install the PyTorch build for your GPU first. The release wheel contains device-code images and has no native Python extension. It supports the WanGP stacks below.

| Platform | Python and PyTorch | Native images |
|---|---|---|
| Windows or Linux x86-64, NVIDIA | Python 3.10 / PyTorch 2.7.1 CUDA 12.8, or Python 3.11 / PyTorch 2.10 CUDA 13 | SM75, SM80, SM86, SM87, SM89, SM90, SM100, SM120, SM121 |
| Windows or Linux x86-64, AMD | Python 3.12 / PyTorch 2.13 ROCm 10 (TheRock) | gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201; Instinct gfx908, gfx90a, gfx942, gfx950 |

If migrating from SpargeAttention, uninstall its distribution before installing SparseGP. Both provide the `spas_sage_attn` import.

```shell
python -m pip uninstall -y spas-sage-attn
python -m pip install https://github.com/deepbeepmeep/SparseGP/releases/download/v1.0.0/sparsegp-1.0.0-py3-none-any.whl
```

The CUDA and HIP paths run without Triton, including the packed Ada and Blackwell router shipped as device code. If a native image cannot load, SparseGP uses its optimized tiled INT8 Triton kernel when the installed compiler supports the GPU. Blackwell requires Triton 3.3 or later for this fallback. On Turing and RDNA 2 the fallback uses packed four-byte integer dot products. Check the actual selection with `backend_for(torch.device("cuda", 0))`. CUDA and HIP both use PyTorch's `cuda` device name.

The source tree includes the compiled images.

```shell
git clone https://github.com/deepbeepmeep/SparseGP
cd SparseGP
python -m pip install .
```

Use `tools/build_images.py` to rebuild the native CUDA images with NVCC or the HIP images with ROCm Clang and rocWMMA. Rebuild the packed matmuls with `sparsegp/packed_cuda/build.py` and their router images with `python tools/build_packed_router.py 89 120` using Triton 3.6.

## Usage

```python
import torch
from sparsegp import backend_for, sol3_attention, sparge_attention

device = torch.device("cuda", 0)
print(backend_for(device))

with torch.inference_mode():
    q, k, v = [torch.randn(1, 4097, 4, 128, device=device, dtype=torch.float16) for _ in range(3)]
    video = sol3_attention(q, k, v, prefix_tokens=37)

    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    # Each entry controls 128 queries by 64 keys, including boundary tiles.
    keep = torch.ones(1, 4, 33, 65, dtype=torch.bool, device=device)
    keep[:, :, :, 1::2] = False
    sparse = sparge_attention(qh, kh, vh, keep)
```

WanGP uses a quality-oriented routing threshold of `tau=0.8`. The standalone Python function defaults to `tau=1.9`. Lower thresholds retain more full key blocks. In WanGP select **Sol3** in **Configuration → Performance → Attention Type**, or select it for one generation in its attention override. Conditioning and VAE decoding use dense attention. The precision option runs the entire first denoising step and the first transformer block on subsequent steps in the available default dense mode. FlashVSR uses SparseGP for its existing Sparge block maps.

## How it works

The router pools groups of 64 tokens. Block relevance selects full blocks. Estimated approximation error, using key variation and value dispersion, selects mixed blocks. Token scores within the mixed blocks are compared with a threshold derived from that query block's mixed-token score distribution. Neighboring blocks and requested conditioning keys remain exact. Selected and representative terms participate in one softmax, with the valid tail length included in the representative mass.

The portable CUDA/HIP kernel uses warp or wave matrix instructions for INT8 QK and FP16 PV, with INT32 and FP32 accumulators respectively. It skips empty key tiles before reading keys and values, and accumulates the output with an online softmax. Ada/Blackwell also have a packed BF16-input path for 128 channels with FP8 PV. The Triton fallback skips empty tiles, implements the same token masks and representative weighting, and JIT-compiles for the active GPU. Sparge's portable path accumulates every selected PV term. The original CUDA implementation's optional additional PV pruning is specific to that implementation.

Run `python tools/validate_gpu.py --target gfx1201 --triton` (substitute the actual architecture) to check masked values, mixed blocks, current-stream operation, CUDA/HIP graph replay and timing. The public release notes distinguish hardware execution from cross-compilation.

## License and attribution

The original SpargeAttention source and the new portable SparseGP kernels use the [Apache License 2.0](LICENSE). The packed kernels retain the [WanGP Community License 2.0](sparsegp/packed_cuda/LICENSE.txt). See [NOTICE](NOTICE) for provenance. Please cite the original [SpargeAttention](README_SpargeAttention.md) and [Sol³](https://github.com/FofGofx/sol3) papers when using their methods.
