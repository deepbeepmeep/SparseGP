# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Sol sparse attention as CUDA kernels for GeForce RTX 50xx (routing rule of NVlabs Sana's Sol-Attn, Apache-2.0).

    prepared = prepare(k, v)                     # k is not read afterwards, and v neither: both can be released
    (or allocate, pack_values, pack_keys, finish when the projections exist one at a time)
    out = attention(q, prepared, tau=1.0, ...)   # out may be q

q, k, v: BF16 [1, tokens, heads, 128] (strided views of a fused projection are read in place). Two precisions:
  - default: INT8 scores, E4M3 values: the precision of the Triton INT8 path within about 0.2 point of error;
  - fp4=True: NVFP4 scores (4-bit codes), faster and less precise.
With either precision, mixed=True routes individual keys within mixed blocks. The prepared state borrows the original
keys. Staged value packing allows the value projection to be released before the key projection.
sol_attn.cu and sol_pack.cu hold the kernels and describe their data; build.py rebuilds the images of ptx/.
"""
import ctypes
from functools import lru_cache
from pathlib import Path


import torch

GROUPED_QUERY = True
BLOCK, GROUP, DIM = 64, 16, 128
_K8_RECORD, _K4_RECORD, _C_RECORD = 16640, 12816, 6208                    # as in the kernels
_SHARED = {False: 2 * _K8_RECORD + _C_RECORD + 8480, True: 2 * _K4_RECORD + _C_RECORD + 8480}
_MAX_LAUNCH = 1 << 37  # query x key pairs of one launch: bounds the time of one driver submission
_IMAGES = Path(__file__).with_name("images")


@lru_cache(None)
def _capability(index):
    return torch.cuda.get_device_capability(index)


def supported(device, dtype=torch.bfloat16):
    """Packed BF16-input kernels on Ada and GeForce Blackwell, outside autograd and compilation."""
    index = device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else None)
    return (device.type == "cuda" and torch.version.hip is None and dtype == torch.bfloat16
            and _capability(index) in ((8, 9), (12, 0))
            and not torch.is_grad_enabled() and not torch.compiler.is_compiling())


def usable(q, k, v):
    """True when prepare(k, v) and attention(q, ...) can read these tensors in place."""
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]
            or not k.size(2) or not q.size(2) or q.size(2) % k.size(2) or q.shape[0] != 1 or q.shape[3] != DIM or q.shape[1] < BLOCK * GROUP):
        return False
    if any(x.dtype != torch.bfloat16 or x.stride(3) != 1 or max(x.stride(1), x.stride(2)) * 2 >= 2**31 for x in (q, k, v)):
        return False
    return q.data_ptr() % 16 == 0 and q.stride(1) % 8 == 0 and q.stride(2) % 8 == 0  # query rows are copied 16 bytes at a time


@lru_cache(None)
def _kernels(device_index, image, names):
    from .. import native
    device = torch.device("cuda", device_index)
    if image == "pack_sm89" and _capability(device_index) == (12,0):
        image = "pack_sm120a"
    module = native.load_image(device, _IMAGES / f"{image}.cubin")
    def launcher(name):
        module.function(name)
        def launch(grid, block, args, *, stream, shared_mem=0):
            # Both the Sol CUDA ABI and the AOT router take their floating-point scalars as doubles.
            values = [ctypes.c_double(v) if isinstance(v, float) else v for v in args]
            module.launch(name, grid, block, values, device, shared_mem)
        return launch
    return {name:launcher(name) for name in names}


def allocate(tokens, heads, device, *, fp4=False, mixed=False):
    """The storage of one attention call's packed keys and values. pack_values and pack_keys fill it in any order, each from a
    projection, then finish gives what attention reads. Mixed routing keeps a reference to the keys for token selection."""
    blocks = (tokens + BLOCK - 1) // BLOCK
    return {"kv": torch.empty((heads, blocks, _K4_RECORD if fp4 else _K8_RECORD), dtype=torch.int8, device=device),
            "c": torch.zeros((heads, (blocks + GROUP - 1) // GROUP, _C_RECORD), dtype=torch.int8, device=device),
            "kc": torch.empty((heads, blocks, DIM), dtype=torch.float32, device=device),
            "peak": torch.zeros((heads, DIM), dtype=torch.float32, device=device), "fp4": fp4, "tokens": tokens, "mixed": mixed}


def _launch(state, x, names, extra=()):
    heads, blocks, record = state["kv"].shape
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    kernels = _kernels(index, "pack_fp4_sm120a" if names == ("sol_pack_k4",) else "pack_sm89", ("sol_pack_k", "sol_v_peak", "sol_pack_v") if names != ("sol_pack_k4",) else names)
    with torch.cuda.device(index):
        for name in names:
            source = [x, state["peak"]] if name != "sol_v_peak" and "v" in name else [x]
            targets = {"sol_v_peak": [state["peak"]], "sol_pack_v": [state["kv"], state["c"]]}.get(name, [state["kv"], state["c"], state["kc"]])
            tail = (record, record - DIM * BLOCK) if name == "sol_pack_v" else ()
            kernels[name]((blocks, heads, 1), (DIM, 1, 1), [*source, *targets, state["tokens"], heads, blocks, x.stride(1) * 2, x.stride(2) * 2, *tail],
                          stream=torch.cuda.current_stream())


def pack_values(state, v):
    _launch(state, v, ("sol_v_peak", "sol_pack_v"))
    if state["mixed"]:
        from .packed_routing import pool
        state["disp"] = pool(v, "value")


def pack_keys(state, k):
    _launch(state, k, ("sol_pack_k4",) if state["fp4"] else ("sol_pack_k",))
    if state["mixed"]:
        from .packed_routing import pool
        state["keys"] = k
        state["kvar"] = pool(k, "key")


def finish(state):
    kc = state.pop("kc")
    if state["mixed"]:
        kc = kc.bfloat16()
        var, mean = torch.var_mean(kc.float(), dim=1, correction=0)
        routing = state["keys"], kc, state["kvar"], state["disp"]
        return state["kv"], state["c"], state["peak"].div_(2.25), mean, var, state["fp4"], routing
    mean = kc.mean(1)
    return state["kv"], state["c"], state["peak"].div_(2.25), mean, kc.sub_(mean[:, None]).square_().mean(1), state["fp4"]


def prepare(k, v, *, fp4=False, mixed=False):
    """Pack the keys and values. Mixed routing borrows k for the lifetime of the prepared state."""
    state = allocate(k.shape[1], k.shape[2], k.device, fp4=fp4, mixed=mixed)
    pack_values(state, v)
    pack_keys(state, k)
    return finish(state)


def attention(q, prepared, *, tau=1.0, scale=None, sink_tokens=0, sink_start=None, out=None):
    """Sol attention of q over prepared = prepare(k, v). The key blocks holding the sink tokens are attended exactly by every
    query (by default the last sink_tokens tokens, else from sink_start). out may be q: a query block is read before it is written."""
    kv, c, vscale, mean, var, fp4 = prepared[:6]
    _, tokens, heads, dim = q.shape
    kv_heads = kv.size(0)
    if not kv_heads or not heads or heads % kv_heads:
        raise ValueError("The query head count must be a positive multiple of the key/value head count")
    blocks = (tokens + BLOCK - 1) // BLOCK
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    first_sink = tokens - sink_tokens if sink_start is None else sink_start
    sinks = (first_sink // BLOCK, (first_sink + sink_tokens + BLOCK - 1) // BLOCK) if sink_tokens else (0, 0)
    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    precision = "fp4" if fp4 else "i8"
    scale = dim ** -0.5 if scale is None else float(scale)
    route = []
    if len(prepared) == 7:
        from .packed_routing import select
        route.append(select(q, *prepared[6], mean, var, scale, tau, sinks))
        precision += "_mixed"
    name = f"sol_attn_{precision}"
    arch = "sm89" if _capability(index) == (8, 9) else "sm120a"
    kernel = _kernels(index, f"attn_{precision}_{arch}", (name,))[name]
    args = [q, kv, c, mean, var, vscale, out, *route, scale, float(tau), *sinks, 0, 0, tokens, heads, blocks,
            q.stride(1) * 2, q.stride(2) * 2, out.stride(1) * 2, out.stride(2) * 2]
    per_launch = max(1, min(blocks, _MAX_LAUNCH // (BLOCK * tokens * heads)))
    stream = torch.cuda.current_stream()
    with torch.cuda.device(index):
        for first in range(0, blocks, per_launch):
            kernel((min(per_launch, blocks - first), kv_heads, heads // kv_heads), (128, 1, 1), [*args, first], shared_mem=_SHARED[fp4], stream=stream)
    torch.autograd.graph.increment_version(out)
    return out


__all__ = ["allocate", "attention", "finish", "pack_keys", "pack_values", "prepare", "supported", "usable"]
