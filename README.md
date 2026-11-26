# SparseGP

**Sparse attention for the GPU Poor.** SparseGP brings Sol³ mixed-block routing with INT8 attention and SpargeAttention block masks to [WanGP](https://github.com/deepbeepmeep/Wan2GP). It ships GPU machine code for GeForce RTX 20–50 series and Radeon RDNA 3/4. The same wheel runs on Windows and Linux.

The fork is based on [SpargeAttention](https://github.com/thu-ml/SpargeAttn). Its original documentation, paper and citation are preserved in [README_SpargeAttention.md](README_SpargeAttention.md). The hierarchical routing method comes from [Sol³ by Fan and coauthors](https://github.com/FofGofx/sol3). WanGP contributed the packed INT8 and NVFP4 paths.

## Attention modes

| Mode | Inputs | Behavior |
|---|---|---|
| `sol3_attention(q, k, v)` | NHD, bidirectional self-attention, FP16 or BF16, head widths 64 and 128, at least 1,024 tokens | INT8 scores. Important blocks keep all keys. Mixed blocks keep selected keys. Each remaining block contributes its mean key and value with the mass of its remaining tokens. |
| `sol3_attention(..., precision="fp4")` | SM120, eligible BF16 width-128 inputs, Blackwell-capable CUDA driver | NVFP4 scores and FP8 values with the same routing. INT8 is the default. |
| `sparge_attention(q, k, v, block_map)` | HND, boolean or 0/1 block map | Attend to the keys allowed by the map. Supports causal masking, rectangular attention and partial final blocks. An empty row returns zero. |
| `spas_sage2_attn_meansim_topk_cuda(...)` | Original SpargeAttention layout and selection arguments | Builds the block map from pooled scores and within-block similarity. The CDF variant is also available. |

For Sol³, a protected prefix or suffix receives dense attention and its key blocks remain exact. Short sequences and incompatible shapes use dense attention. A dense Sparge call uses [SageGP](https://github.com/deepbeepmeep/SageGP) when its kernel supports the call. PyTorch SDPA handles other dense requests, including explicit token masks. With an explicit block map SparseGP preserves the map even if the fast operator cannot serve the shape.

### Krea 2 Turbo grouped-query attention on RTX 5090

Krea 2 shares 12 K/V heads across 48 query heads. The following Windows runs used the same checkpoint, fox prompt, seed 42, eight denoising steps, CUDA graphs and WanGP Sol3 tau 0.8. Times are median wall time for three warm full image generations per mode. SSIM compares each image to dense PyTorch SDPA at its resolution. A value of 1 means identical images.

| Resolution | Precise first step and first block | Sage2GP time | Sol3 INT8 time | Sage2GP SSIM | Sol3 SSIM |
|---|---|---:|---:|---:|---:|
| 512 × 512 | Off | 3.98 s | 3.93 s | 0.991 | 0.905 |
| 512 × 512 | On | 3.66 s | 3.57 s | 0.991 | 0.948 |
| 1024 × 1024 | Off | 5.22 s | 4.81 s | 0.942 | 0.739 |
| 1024 × 1024 | On | 4.55 s | 4.37 s | 0.942 | 0.915 |

Each row compares the modes in the same process. The two precision settings ran in separate sessions and their timings varied. At 1024 pixels, Sol3 without the precise first step changed the direction the fox was walking. Enabling the option preserved its direction and framing. The Sage2GP runs used WanGP's staged per-warp INT8 Q/K and FP8 V kernel for grouped-query attention. [Side-by-side images](https://github.com/deepbeepmeep/SparseGP/releases/download/v1.2.0/krea2-gqa-comparison.jpg) · [Individual runs and quality metrics](https://github.com/deepbeepmeep/SparseGP/releases/download/v1.2.0/krea2-gqa-benchmarks.json)

### Sol clip measurements from v1.0.0

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

### Sparge block maps on RTX 5090

Version 1.1.0 compiles the original SpargeAttention CUDA kernels and adds fused block-map preparation, direct strided output, and zeroing of empty rows in the same launch. A 64-query Blackwell variant shares each 128-query routing decision across two CTAs to occupy more SMs during small head-group launches. The original 128-query variant handles larger launches.

| BF16, eight heads, width 128 | Blocks kept | Original Windows Sparge wheel | SparseGP 1.0.0 | SparseGP 1.1.0 |
|---|---:|---:|---:|---:|
| 2,048 tokens | 25% | 0.302 ms | 0.367 ms | 0.242 ms |
| 8,192 tokens | 12.5% | 0.323 ms | 1.917 ms | 0.235 ms |
| 8,192 tokens | 50% | 0.471 ms | 6.867 ms | 0.384 ms |
| 16,384 tokens | 25% | 0.890 ms | 13.426 ms | 0.675 ms |

These are attention-call timings including projection packing and map preparation, on the same random tensors and masks in one Windows process. Each cell is a median over twelve groups of five calls after warmup. On the four rows above the original wheel and 1.1.0 both have about 3.7–3.9% relative L2 error against FP32 masked SDPA on 64 sampled queries. Version 1.0.0 had about 0.9% with its FP16 probability–value product. These figures measure kernel output on synthetic inputs. The earlier clip table measures Sol3. [Raw timings and numerical errors](https://github.com/deepbeepmeep/SparseGP/releases/download/v1.1.0/sparge-benchmarks.json)

## Installation

Install the PyTorch build for your GPU first. The release wheel contains device-code images and has no native Python extension. It supports the WanGP stacks below.

| Platform | Python and PyTorch | Native images |
|---|---|---|
| Windows or Linux x86-64, NVIDIA | Python 3.10 / PyTorch 2.7.1 CUDA 12.8, or Python 3.11 / PyTorch 2.10 CUDA 13 | SM75, SM80, SM86, SM87, SM89, SM90, SM100, SM120, SM121 |
| Windows or Linux x86-64, AMD | Python 3.12 / PyTorch 2.13 ROCm 10 (TheRock) | gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201; Instinct gfx908, gfx90a, gfx942, gfx950 |

If migrating from SpargeAttention, uninstall its distribution before installing SparseGP. Both provide the `spas_sage_attn` import.

```shell
python -m pip uninstall -y spas-sage-attn
python -m pip install https://github.com/deepbeepmeep/SparseGP/releases/download/v1.2.0/sparsegp-1.2.0-py3-none-any.whl
```

The CUDA and HIP paths run without Triton, including the packed Ada and Blackwell router shipped as device code. If a native image cannot load, SparseGP uses its optimized tiled INT8 Triton kernel when the installed compiler supports the GPU. Blackwell requires Triton 3.3 or later for this fallback. On Turing and RDNA 2 the fallback uses packed four-byte integer dot products. Check the actual selection with `backend_for(torch.device("cuda", 0))`. CUDA and HIP both use PyTorch's `cuda` device name.

The source tree includes the compiled images.

```shell
git clone https://github.com/deepbeepmeep/SparseGP
cd SparseGP
python -m pip install .
```

Build the Sparge tensor-core images with `python tools/build_sparge.py 80 86 87 89 90 100 120 121` and NVCC (use `--ccbin` for the host compiler if needed). Run `python tools/build_aot.py TARGET ...` with Triton 3.6 at build time for the architecture-specific mixed-block, block-list and preparation kernels. Targets use the CUDA capability (for example `120`) or the AMD ISA (for example `gfx1201`). Rebuild the packed router intermediates with `python tools/build_packed_router.py 89 120`, then assemble packed Sol, Sol3 and their router into cubins with `python sparsegp/packed_cuda/build.py --regenerate-ptx`. The installed native paths load only cubin or HSACO machine code.


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

Sol3 accepts grouped-query self-attention in NHD layout. The query head count must be a multiple of the K/V head count. Krea 2, for example, shares each of its 12 K/V heads among four of its 48 query heads. SparseGP packs those 12 heads once and routes each query head separately. The packed Ada and Blackwell kernels process all query heads in the same launch. Other supported devices share the same K/V storage across groups of query heads. Protected prefix or suffix queries use dense grouped-query attention.

WanGP uses a quality-oriented routing threshold of `tau=0.8`. The standalone Python function defaults to `tau=1.9`. Lower thresholds retain more full key blocks. In WanGP select **Sol3** in **Configuration → Performance → Attention Type**, or select it for one generation in its attention override. Conditioning and VAE decoding use dense attention. The precision option runs the entire first denoising step and the first transformer block on subsequent steps in the available default dense mode. FlashVSR uses SparseGP for its existing Sparge block maps.

## How it works

The router pools groups of 64 tokens. Block relevance selects full blocks. Estimated approximation error, using key variation and value dispersion, selects mixed blocks. Token scores within the mixed blocks are compared with a threshold derived from that query block's mixed-token score distribution. Neighboring blocks and requested conditioning keys remain exact. Selected and representative terms participate in one softmax, with the valid tail length included in the representative mass.

Hopper also has the original 64-query by 128-key WGMMA kernel. For standard 128-query by 64-key block maps, SM80 and newer run the original SpargeAttention tensor-core code compiled for the selected GPU. Ampere uses FP16 values. Ada and newer use FP8 values and fuse value scales. The compiled map builder accepts strided block maps and the attention kernel clears empty rows in place. Other supported geometries, Turing and AMD use target-specific compiled kernels that iterate selected key indices. Sol3 has separate mixed-token kernels, including the packed BF16-input INT8 path on Ada/Blackwell and NVFP4 on SM120. The JIT fallback is used when a compatible machine image cannot be loaded.

Run `python tools/validate_gpu.py --target gfx1201 --triton` (substitute the actual architecture) to check masked values, mixed blocks, current-stream operation, CUDA/HIP graph replay and timing. The public release notes distinguish hardware execution from cross-compilation.

## License and attribution

The original SpargeAttention source and the architecture-specific SparseGP kernels use the [Apache License 2.0](LICENSE). The packed kernels retain the [WanGP Community License 2.0](sparsegp/packed_cuda/LICENSE.txt). See [NOTICE](NOTICE) for provenance. Please cite the original [SpargeAttention](README_SpargeAttention.md) and [Sol³](https://github.com/FofGofx/sol3) papers when using their methods.
