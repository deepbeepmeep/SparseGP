# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Launch the shipped Ada and Blackwell routing images on the caller's current stream."""

from functools import lru_cache
import json
import torch
from . import _IMAGES, _kernels


@lru_cache(None)
def _kernel(index, name):
    major, minor = torch.cuda.get_device_capability(index)
    arch = major * 10 + minor
    entry, shared, scratch = json.loads((_IMAGES / f"route_{arch}.json").read_text())[name]
    kernel = _kernels(index, f"route_{arch}_{name}", (entry,))[entry]
    return kernel, shared, scratch


def _launch(name, grid, arguments):
    device = arguments[0].device
    kernel, shared, scratch = _kernel(device.index, name)
    # The emitted ABI can include unused scratch pointers (both scratch sizes are zero).
    kernel(
        grid,
        (128, 1, 1),
        [*arguments, *([arguments[0]] * scratch)],
        shared_mem=shared,
        stream=torch.cuda.current_stream(device),
    )


def _aligned(x):
    if x.data_ptr() % 16 or x.stride(1) % 16 or x.stride(2) % 16:
        return x.clone(memory_format=torch.contiguous_format)
    return x


def pool(x, kind):
    x = _aligned(x)
    _, tokens, heads, _ = x.shape
    blocks = (tokens + 63) // 64
    shape = (heads, blocks) if kind == "value" else (heads, blocks, 128)
    out = torch.empty(shape, dtype=torch.float32, device=x.device)
    _launch(kind, (blocks, heads, 1), [x, out, tokens, blocks, x.stride(1), x.stride(2)])
    return out


def select(q, keys, kc, kvar, disp, mean, between, scale, tau, sinks):
    heads, n = kc.shape[:2]
    tokens = q.shape[1]
    keys = _aligned(keys)
    qb = pool(q, "query")
    flags = torch.empty((heads, n, n), device=q.device, dtype=torch.int8)
    _launch(
        "classify",
        ((n + 15) // 16, (n + 63) // 64, heads),
        [qb, kc, kvar, disp, mean, between, kvar.mean(1), disp.mean(1), flags, n, scale, float(tau), *sinks],
    )
    nt = (tokens + 255) // 256
    partial = torch.empty((heads, n, nt, 3), device=q.device, dtype=torch.float32)
    cut = torch.empty((heads, n), device=q.device, dtype=torch.float32)
    bits = torch.empty_like(flags, dtype=torch.int64)
    grid = ((n + 31) // 32, nt, heads)
    args = [qb, keys, flags, partial, cut, bits, tokens, n, nt, keys.stride(1), keys.stride(2), scale]
    _launch("moments", grid, args)
    _launch("cut", (heads * n, 1, 1), [partial, cut, nt])
    _launch("select", grid, args)
    return bits
