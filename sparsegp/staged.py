"""Sparge projection staging for WanGP. Map geometry is 128 query rows and 64 key rows."""

import torch
from .core import _backend, backend_for
from .routing import pad, round_up


def staged_settings(device, head_dim):
    return {"block_q": 128, "block_k": 64} if head_dim in (64, 128) and backend_for(device) else None


def _rows(x):
    b, t, h, d = x.shape
    return x.transpose(1, 2).reshape(b * h, t, d).to(torch.float16)


def staged_quantize_q(q, settings):
    with torch.cuda.device(q.device):
        return _backend(q.device, "auto").pack(pad(_rows(q), round_up(q.size(1), 64)))


def staged_quantize_k(k_list, settings):
    k = k_list.pop()
    tokens = k.size(1)
    with torch.cuda.device(k.device):
        rows = _rows(k)
        quant, scale = _backend(k.device, "auto").pack(pad(rows - rows.mean(1, keepdim=True), round_up(tokens, 64)))
    return quant, scale, tokens


def staged_quantize_v(v_list, settings):
    v = v_list.pop()
    return pad(_rows(v), round_up(v.size(1), 64))


def staged_attention(out, quantized_list, settings, attn_mask=None):
    (q, qs), (k, ks, keys), v = quantized_list
    quantized_list.clear()
    b, t, h, d = out.shape
    keep = attn_mask.expand(b, h, (t + 127) // 128, (keys + 63) // 64).reshape(
        b * h, (t + 127) // 128, (keys + 63) // 64
    )
    bits = (-keep.bool().int()).unsqueeze(-1).expand(*keep.shape, 2).contiguous()
    with torch.cuda.device(out.device):
        value = _backend(out.device, "auto").attend(q, k, v, qs, ks, bits, keys, 128, d**-0.5, False, False)
    out.copy_(value[:, :t].view(b, h, t, d).transpose(1, 2))
    return out
