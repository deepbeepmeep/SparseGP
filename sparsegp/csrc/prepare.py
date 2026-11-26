# Copyright (c) 2026 DeepBeepMeep. Projection packing is adapted from WanGP.
# See packed_cuda/LICENSE.txt and NOTICE.
"""Ahead-of-time projection packing for Sparge, adapted from WanGP (see NOTICE)."""

import triton
import triton.language as tl


@triton.jit
def _quant_block_nhd(
    X,
    Mean,
    Xq,
    Scale,
    tokens,
    heads,
    sxb,
    sxn,
    sxh,
    sqb,
    sqn,
    sqh,
    ssb,
    ssh,
    D: tl.constexpr,
    BLK: tl.constexpr,
    SUB_MEAN: tl.constexpr,
):
    """qk_quantize's arithmetic (the mean subtracted in x's dtype, block scale max|x| / 127 + 1e-7, rounding half away from zero) on
    strided (batch, tokens, heads, D) tensors; Mean is (batch, 1, heads, D) contiguous in x's dtype."""
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    rows = block * BLK + tl.arange(0, BLK)
    cols = tl.arange(0, D)
    inside = rows[:, None] < tokens
    x = tl.load(X + batch * sxb + head * sxh + rows[:, None] * sxn + cols[None, :], inside, 0.0)
    if SUB_MEAN:
        x -= tl.load(Mean + (batch * heads + head) * D + cols)[None, :]
        x = tl.where(inside, x, 0.0)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 127.0
    scale += 1e-7
    q = x / scale
    q += 0.5 * tl.where(q >= 0, 1, -1)
    tl.store(Xq + batch * sqb + head * sqh + rows[:, None] * sqn + cols[None, :], q.to(tl.int8), inside)
    tl.store(Scale + batch * ssb + head * ssh + block, scale)


@triton.jit
def _v_fp8_nhd(
    V, Low, High, V8, Scale, tokens, heads, svb, svn, svh, pad_len, scale_max, D: tl.constexpr, BLOCK: tl.constexpr
):
    """scale_fuse_quant_cuda's FP8 V straight from (batch, tokens, heads, D), without its 16-bit transposed copy: per channel the scale
    amax / scale_max (amax from the channel's min and max over the tokens), the values times scale_max / amax rounded to e4m3, stored
    transposed (batch, D, heads, padded tokens) with the fp8 MMA permutation inside every 16 tokens, zeros beyond the tokens. The
    divisions are Triton's fp32 division, the instruction the CUDA kernel uses under fast math. The first block stores the scales."""
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    channel = (batch * heads + head) * D + cols
    amax = tl.maximum(-tl.load(Low + channel).to(tl.float32), tl.load(High + channel).to(tl.float32))
    if block == 0:
        tl.store(Scale + channel, amax / scale_max)
    x = tl.load(V + batch * svb + head * svh + rows[:, None] * svn + cols[None, :], rows[:, None] < tokens, 0.0).to(
        tl.float32
    )
    x = x * tl.where(amax > 0, scale_max / amax, 0.0)[None, :]
    t = rows % 16
    col = (rows // 16) * 16 + (t // 8) * 2 + ((t // 2) % 4) * 4 + t % 2
    tl.store(V8 + ((batch * D + cols)[None, :] * heads + head) * pad_len + col[:, None], x.to(tl.float8e4nv))


@triton.jit
def _lookup(
    Keep,
    Lut,
    Count,
    rows,
    cols,
    sh,
    sb,
    sr,
    sc,
    heads,
    qb,
    kb,
    CAUSAL: tl.constexpr,
    DELTA: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    index = (batch * heads + head) * rows + row
    start = Keep + batch * sb + head * sh + row * sr
    lanes = tl.arange(0, CHUNK)
    used = 0
    previous = 0
    for tile in range(tl.cdiv(cols, CHUNK)):
        c = tile * CHUNK + lanes
        keep = tl.load(start + c * sc, c < cols, 0) != 0
        if CAUSAL:
            keep = keep & (c * kb < (row + 1) * qb)
        pos = tl.cumsum(keep.to(tl.int32), 0) - 1 + used
        if DELTA:
            last = tl.associative_scan(tl.where(keep, c, previous), 0, _max)
            prior = tl.gather(last, tl.maximum(lanes - 1, 0), axis=0)
            value = c - tl.where(lanes == 0, previous, prior)
            previous = tl.max(last, 0)
        else:
            value = c
        tl.store(Lut + index * cols + pos, value, keep)
        used += tl.sum(keep.to(tl.int32), 0)
    tl.store(Count + index, used)


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)
