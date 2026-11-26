"""Public API. Views use NHD for Sol³ and HND for the Sparge compatibility call."""

from functools import lru_cache
from importlib.util import find_spec
import torch
import torch.nn.functional as F
from . import native
from .routing import pad, round_up, sol_route


def backend_for(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return _device_backend(torch.device("cuda", torch.cuda.current_device()) if device.index is None else device)


@lru_cache(None)
def _device_backend(device):
    if native.image_path(device).is_file():
        try:
            native.module(device)
        except (OSError, native.ImageUnavailable):
            pass
        else:
            return "hip" if torch.version.hip else "cuda"
    return "triton" if triton_supported(device) else None


@lru_cache(None)
def triton_supported(device):
    if find_spec("triton") is None:
        return False
    try:
        import triton
        from triton.backends import backends
    except (ImportError, OSError):
        return False

    if torch.version.hip:
        return "amd" in backends
    capability = torch.cuda.get_device_capability(device)
    version = tuple(int(n) for n in triton.__version__.split(".")[:2])
    # LLVM in Triton 3.2 aborts the process when lowering warp reductions for SM120.
    return "nvidia" in backends and capability >= (7, 5) and (capability < (10, 0) or version >= (3, 3))


def _backend(device, requested):
    name = backend_for(device) if requested == "auto" else requested
    if name == "triton":
        if not triton_supported(device):
            raise RuntimeError(f"The installed Triton compiler does not support {native.target(device)}")
        from . import triton_backend

        return triton_backend
    if name in ("cuda", "hip") and name == backend_for(device):
        return native
    raise ValueError(f"No {name} sparse attention kernel on {device}")


def dense_attention(q, k, v, *, scale=None, causal=False, mask=None, dropout_p=0.0):
    """Dense HND call, preserving scale, masking, causality and grouped-query heads."""
    if (
        q.is_cuda
        and not (torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)))
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype
        and not torch.version.hip
        and q.dtype in (torch.float16, torch.bfloat16)
        and (not causal or q.size(2) == k.size(2))
        and mask is None
        and not dropout_p
        and q.size(-1) in (64, 128)
        and q.size(-1) == v.size(-1)
        and native.target(q.device)
        in {"sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90", "sm_100", "sm_120", "sm_121"}
    ):
        try:
            from sagegp import sageattn
        except ImportError:
            pass
        else:
            return sageattn(q, k, v, tensor_layout="HND", is_causal=causal, sm_scale=scale)
    if mask is not None and causal:
        rows = torch.arange(q.size(2), device=q.device)
        cols = torch.arange(k.size(2), device=q.device)
        result = []
        for start in range(0, q.size(2), 256):
            visible = cols <= rows[start : start + 256, None]
            weights = mask if mask.ndim < 2 or mask.size(-2) == 1 else mask[..., start : start + 256, :]
            combined = (
                visible & weights if weights.dtype == torch.bool else torch.where(visible, weights, -torch.inf)
            )
            result.append(
                dense_attention(q[:, :, start : start + 256], k, v, scale=scale, mask=combined, dropout_p=dropout_p)
            )
        return torch.cat(result, dim=2) if result else v.new_empty((*q.shape[:-1], v.size(-1)))
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
        enable_gqa=q.size(1) != k.size(1),
    )


def _eligible(q, k, v):
    return (
        q.ndim == 4
        and not (torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)))
        and k.ndim == v.ndim == 4
        and q.shape[:2] == k.shape[:2]
        and k.shape == v.shape
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] in (64, 128)
        and min(q.size(2), k.size(2)) > 0
        and q.dtype == k.dtype == v.dtype
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.device == k.device == v.device
        and backend_for(q.device) is not None
    )


def _execute(engine, q, k, v, bits, keys, block_q, scale, reps, causal):
    qi, qs = engine.pack(pad(q, round_up(q.size(1), 64)))
    ki, ks = engine.pack(k)
    out = engine.attend(qi, ki, v, qs, ks, bits, keys, block_q, scale, reps, causal)
    return out[:, : q.size(1)]


@torch.compiler.disable
def sol3_attention(
    q, k, v, *, scale=None, tau=1.9, prefix_tokens=0, suffix_tokens=0, backend="auto", precision="int8"
):
    """Bidirectional NHD attention. Protected prefix/suffix queries run dense and their keys remain exact."""
    if precision not in ("int8", "fp4"):
        raise ValueError("precision must be int8 or fp4")
    if prefix_tokens and suffix_tokens:
        raise ValueError("Specify a protected prefix or suffix")
    tq = q.size(1)
    length = prefix_tokens or suffix_tokens
    if not (0 <= prefix_tokens <= tq and 0 <= suffix_tokens <= tq):
        raise ValueError("Conditioning length exceeds query length")
    start = 0 if prefix_tokens else tq - suffix_tokens
    qh, kh, vh = [x.transpose(1, 2) for x in (q, k, v)]
    if not _eligible(qh, kh, vh) or q.shape != k.shape or tq < 1024 or length == tq:
        if precision == "fp4":
            raise ValueError("FP4 requires eligible Blackwell BF16 self-attention")
        return dense_attention(qh, kh, vh, scale=scale).transpose(1, 2)
    scale = q.size(-1) ** -0.5 if scale is None else float(scale)
    result = torch.empty_like(q, memory_format=torch.contiguous_format)
    with torch.cuda.device(q.device):
        for b in range(q.size(0)):
            tensors = tuple(x[b : b + 1] for x in (q, k, v))
            if backend in ("auto", "cuda") and backend_for(q.device) == "cuda":
                from . import packed_cuda

                if packed_cuda.supported(q.device, q.dtype) and packed_cuda.usable(*tensors):
                    if precision == "fp4" and native.target(q.device) != "sm_120":
                        raise ValueError("FP4 requires SM120")
                    state = packed_cuda.prepare(tensors[1], tensors[2], fp4=precision == "fp4", mixed=True)
                    packed_cuda.attention(
                        tensors[0],
                        state,
                        tau=tau,
                        scale=scale,
                        sink_tokens=length,
                        sink_start=start,
                        out=result[b : b + 1],
                    )
                    del state
                    continue
            if precision == "fp4":
                raise ValueError("FP4 requires the SM120 packed CUDA kernel")
            engine = _backend(q.device, backend)
            for h in range(0, q.size(2), 4):
                x, y, z = (t[b, h : h + 4].to(torch.float16).contiguous() for t in (qh, kh, vh))
                bits, kc, vc = sol_route(x, y, z, scale, tau, start, length, engine)
                exact = round_up(tq, 64)
                kbuf = torch.cat((pad(y, exact), pad(kc, round_up(kc.size(1), 64))), 1)
                vbuf = torch.cat((pad(z, exact), pad(vc, round_up(vc.size(1), 64))), 1)
                output = _execute(engine, x, kbuf, vbuf, bits, tq, 64, scale, True, False)
                result[b, :, h : h + 4].copy_(output.transpose(0, 1))
        if length:
            result[:, start : start + length].copy_(
                dense_attention(qh[:, :, start : start + length], kh, vh, scale=scale).transpose(1, 2)
            )
    return result


@torch.compiler.disable
def sparge_attention(
    q,
    k,
    v,
    block_map=None,
    *,
    block_size=(128, 64),
    scale=None,
    causal=False,
    backend="auto",
    dropout_p=0.0,
    smooth_k=True,
    pv_threshold=50.,
):
    """HND block-sparse attention with an explicit [B,H,query blocks,key blocks] keep-map."""
    if block_map is None:
        return dense_attention(q, k, v, scale=scale, causal=causal, dropout_p=dropout_p)
    b, h, tq, d = q.shape
    tk = k.size(2)
    mq, mk = block_size
    if mq not in (64, 128) or mk not in (64, 128):
        raise ValueError("Sparge block sizes must be 64 or 128")
    block_map = block_map.expand(b, h, (tq + mq - 1) // mq, (tk + mk - 1) // mk).bool()
    if not _eligible(q, k, v) or dropout_p:
        # Chunk the query axis so a fallback does not materialize a token-square mask.
        rows = torch.arange(tq, device=q.device)
        cols = torch.arange(tk, device=q.device)
        outputs = []
        for start in range(0, tq, 256):
            rr = rows[start : start + 256]
            mask = block_map[:, :, rr[:, None] // mq, cols[None, :] // mk]
            if causal:
                mask = mask & (cols[None, :] <= rr[:, None])
            outputs.append(
                dense_attention(q[:, :, start : start + 256], k, v, scale=scale, mask=mask, dropout_p=dropout_p)
            )
        return torch.cat(outputs, 2)
    scale = d**-0.5 if scale is None else float(scale)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    with torch.cuda.device(q.device):
        engine = _backend(q.device, backend)
        if engine is native:
            from . import sparge_cuda
            settings = sparge_cuda.settings(q.device,d,(mq,64) if mq==128 else (mq,mk))
            if settings is not None:
                cfg = dict(settings, smooth_k=smooth_k, causal=causal, scale=scale, pv_threshold=pv_threshold)
                keep = block_map if mk==64 or settings["hopper"] else block_map.repeat_interleave(2,-1)[..., :(tk+63)//64]
                values = sparge_cuda.staged_quantize_v([v.transpose(1,2)], cfg)
                keys = sparge_cuda.staged_quantize_k([k.transpose(1,2)], cfg)
                query = sparge_cuda.staged_quantize_q(q.transpose(1,2), cfg)
                sparge_cuda.staged_attention(out.transpose(1,2), [query,keys,values], cfg, keep)
                return out
        for batch in range(b):
            for head in range(0, h, 8):
                x, y, z = (t[batch, head : head + 8].to(torch.float16).contiguous() for t in (q, k, v))
                if smooth_k:
                    y = y - y.mean(1, keepdim=True)
                if engine is native:
                    query,qs = engine.pack(pad(x,round_up(tq,64)))
                    key,ks = engine.pack(pad(y,round_up(tk,64)))
                    keep = block_map[batch:batch+1,head:head+8]
                    value = engine.attend_blocks(query,key,pad(z,round_up(tk,64)),qs,ks,keep,tk,mq,mk,scale,causal)
                    out[batch,head:head+8].copy_(value[:,:tq])
                    continue
                keep = block_map[batch, head : head + 8].repeat_interleave(mk // 64, -1)[..., : (tk + 63) // 64]
                # A Sparge tile always includes every valid key or none. Tail keys
                # are bounded independently by the attention kernel.
                bits = (-keep.int()).unsqueeze(-1).expand(*keep.shape, 2).contiguous()
                rows = round_up(tk, 64)
                out[batch, head : head + 8].copy_(
                    _execute(engine, x, pad(y, rows), pad(z, rows), bits, tk, mq, scale, False, causal)
                )
    return out


def block_sparse_sage2_attn_cuda(
    q,
    k,
    v,
    mask_id=None,
    dropout_p=0.0,
    scale=None,
    smooth_k=True,
    pvthreshd=50,
    attention_sink=False,
    tensor_layout="HND",
    output_dtype=torch.float16,
    return_sparsity=False,
    *,
    block_size=None,
):
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError("Expected HND or NHD")
    if block_size is None:
        block_size = (64, 128) if q.is_cuda and native.target(q.device) == "sm_90" else (128, 64)
    if tensor_layout == "NHD":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = sparge_attention(
        q, k, v, mask_id, block_size=block_size, scale=scale, dropout_p=dropout_p, smooth_k=smooth_k, pv_threshold=pvthreshd
    )
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    if return_sparsity:
        return out, 1 - mask_id.float().mean().item() if mask_id is not None else 0.0
    return out
