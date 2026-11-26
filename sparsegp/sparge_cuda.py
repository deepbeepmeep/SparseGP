"""Projection and block-map operations for the precompiled upstream Sparge tensor cores."""

import ctypes as C
from functools import lru_cache

import torch
from . import native


@lru_cache(None)
def settings(device, width, block_size=(128, 64)):
    if torch.version.hip or width not in (64, 128):
        return None
    arch = native.target(device)
    hopper = arch == "sm_90" and block_size == (64, 128)
    if not hopper and block_size != (128, 64):
        return None
    path = native.image_path(device).parent.parent / ("sparge_" + arch + ("a" if hopper else "") + ".cubin")
    if not path.is_file():
        return None
    try:
        image = native.load_image(device, path)
        image.function(f"sparge_{width}_bf16")
    except (OSError, native.ImageUnavailable):
        return None
    fp8 = torch.cuda.get_device_capability(device) >= (8, 9)
    return dict(
        image=image,
        fp8=fp8,
        hopper=hopper,
        small_tile_sms=(
            torch.cuda.get_device_properties(device).multi_processor_count
            if torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
            else 0
        ),
    )


def _dtype(x):
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Sparge tensor cores require FP16 or BF16 projections")
    return "bf16" if x.dtype == torch.bfloat16 else "fp16"


def _quant(x, kind, cfg, mean=None):
    if x.stride(-1) != 1:
        x = x.contiguous()
    b, t, h, d = x.shape
    block = 128 if kind == "q" else 64
    if cfg["hopper"]:
        block = 64 if kind == "q" else 128
        kind += str(block)
    n = (t + block - 1) // block
    shape = (b, h, t, d) if cfg["hopper"] else x.shape
    quant = torch.empty(shape, device=x.device, dtype=torch.int8)
    if cfg["hopper"]:
        quant = quant.transpose(1, 2)
    scales = torch.empty((b, h, n), device=x.device, dtype=torch.float32)
    native.launch(
        x.device,
        f"{kind}_{d}_{_dtype(x)}",
        (n, h, b),
        (
            x,
            x if mean is None else mean,
            quant,
            scales,
            t,
            h,
            *x.stride()[:3],
            *quant.stride()[:3],
            *scales.stride()[:2],
        ),
    )
    return quant, scales


def staged_quantize_q(q, cfg):
    return _quant(q, "q", cfg)


def staged_quantize_k(k_list, cfg):
    k = k_list.pop()
    if cfg.get("smooth_k", True):
        return _quant(k, "k", cfg, k.mean(1, keepdim=True).contiguous())
    return _quant(k, "kraw", cfg)


def staged_quantize_v(v_list, cfg):
    v = v_list.pop()
    if not cfg["fp8"]:
        return v.to(torch.float16).contiguous(), None
    if v.stride(-1) != 1:
        v = v.contiguous()
    b, t, h, d = v.shape
    padded = (t + 127) // 128 * 128
    low, high = torch.aminmax(v, dim=1)
    packed = torch.empty((b, d, h, padded), dtype=torch.float8_e4m3fn, device=v.device)
    scale = torch.empty((b, h, d), dtype=torch.float32, device=v.device)
    native.launch(
        v.device,
        f"v_{d}_{_dtype(v)}",
        (padded // 64, h, b),
        (v, low, high, packed, scale, t, h, *v.stride()[:3], padded, 448.0 if cfg["hopper"] else 2.25),
    )
    if cfg["hopper"]:
        packed = packed.transpose(1, 2).contiguous()
    return packed, scale


def _tensor_map(image, tensor, rows, columns):
    # cuTensorMapEncodeTiled requires a 64-byte aligned, 128-byte descriptor. The tensor is byte-sized HND.
    storage = C.create_string_buffer(191)
    desc = (C.c_char * 128).from_buffer(storage, (-C.addressof(storage)) % 64)
    u64, u32 = C.c_uint64, C.c_uint32
    shape = (u64 * 4)(*reversed(tensor.shape))
    strides = (u64 * 3)(*reversed(tensor.stride()[:3]))
    swizzle = {64: 2, 128: 3}[columns]
    image.check(
        image.driver.cuTensorMapEncodeTiled(
            C.byref(desc),
            0,
            4,
            C.c_void_p(tensor.data_ptr()),
            shape,
            strides,
            (u32 * 4)(columns, rows, 1, 1),
            (u32 * 4)(1, 1, 1, 1),
            0,
            swizzle,
            0,
            0,
        )
    )
    return desc


def staged_attention(out, projections, cfg, attn_mask):
    (q, qs), (k, ks), (v, vs) = projections
    projections.clear()
    b, t, h, d = q.shape
    n = k.size(1)
    qb, kb = (64, 128) if cfg["hopper"] else (128, 64)
    nq, nk = (t + qb - 1) // qb, (n + kb - 1) // kb
    keep = attn_mask.expand(b, h, nq, nk)
    causal = bool(cfg.get("causal", False))
    lut, count = native.block_indices(keep, qb, kb, causal, delta=True)
    threshold = cfg.get("pv_threshold", 50.0)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(q.device, torch.float32).expand(h).contiguous()
    else:
        threshold = torch.full((h,), float(threshold), dtype=torch.float32, device=q.device)
    dtype = "bf16" if out.dtype == torch.bfloat16 else "f16"
    name = f"sparge_{d}_{dtype}" + ("_causal" if causal else "")
    # Two shorter CTAs cover each map row when a long, head-split launch would leave SMs idle.
    # The same Q scales and routing bits feed both CTAs. Per-head pruning settings retain 128-row groups.
    split = (
        not causal
        and nk >= 128
        and b * h * nq <= cfg["small_tile_sms"]
        and isinstance(cfg.get("pv_threshold", 50.0), (float, int))
        and cfg.get("pv_threshold", 50.0) == 50.0
    )
    rows = 64 if split else 128
    if rows == 64:
        name += "_q64"
    # The upstream kernel writes 128-bit vectors. Projection views can be unaligned.
    dst = (
        out
        if out.data_ptr() % 16 == 0 and out.stride(-1) == 1 and all(i % 8 == 0 for i in out.stride()[:3])
        else torch.empty_like(out, memory_format=torch.contiguous_format)
    )
    strides = (v.stride(0), v.stride(2), v.stride(1)) if vs is not None else v.stride()[:3]
    image = cfg["image"]
    if cfg["hopper"]:
        maps = (
            _tensor_map(image, q.transpose(1, 2), qb, d),
            _tensor_map(image, k.transpose(1, 2), kb, d),
            _tensor_map(image, v, d, kb),
        )
        image.launch(
            name,
            (nq, h, b),
            (128, 1, 1),
            (
                *maps,
                qs,
                ks,
                vs,
                dst,
                lut,
                count,
                threshold,
                dst.stride(0),
                dst.stride(2),
                dst.stride(1),
                t,
                n,
                float(cfg.get("scale", d**-0.5)),
            ),
            q.device,
            (qb + 2 * kb) * d,
        )
    else:
        cfg["image"].launch(
            name,
            ((t + rows - 1) // rows, h, b),
            (32, 8 if not cfg["fp8"] and d == 128 else 4, 1),
            (
                q,
                k,
                v,
                dst,
                lut,
                count,
                threshold,
                qs,
                ks,
                vs if vs is not None else qs,
                t,
                n,
                *strides,
                *dst.stride()[:3],
                float(cfg.get("scale", d**-0.5)),
            ),
            q.device,
            ((rows + 128) if cfg["fp8"] else 320) * d,
        )
    torch.autograd.graph.increment_version(dst)
    if dst is not out:
        out.copy_(dst)
    return out
