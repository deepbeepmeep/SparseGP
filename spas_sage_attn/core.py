"""SpargeAttention's public functions on the native SparseGP backends."""

import torch
import torch.nn.functional as F

from sparsegp import block_sparse_sage2_attn_cuda, dense_attention, sparge_attention
from sparsegp.routing import round_up


def _pool(x, block):
    n = x.size(2)
    rows = F.pad(x, (0, 0, 0, round_up(n, block) - n)).unflatten(2, (-1, block))
    length = (n - torch.arange(rows.size(2), device=x.device) * block).clamp(max=block)
    mean = rows.sum(3, dtype=torch.float32) / length[:, None]
    # The mean of all pairwise cosine similarities is the squared norm
    # of the mean unit vector. Padding contributes neither mass nor norm.
    unit = F.normalize(rows.float(), dim=-1)
    sim = (unit.sum(3) / length[:, None]).square().sum(-1)
    return mean.to(x.dtype), sim


def _meansim(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    smooth_k,
    simthreshd1,
    cdfthreshd,
    topk,
    attention_sink,
    pvthreshd,
    tensor_layout,
    return_sparsity,
):
    if tensor_layout not in ("NHD", "HND"):
        raise ValueError("Expected HND or NHD")
    if tensor_layout == "NHD":
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

    def head_param(value):
        return torch.as_tensor(value, device=q.device, dtype=torch.float32).reshape(1, -1, 1, 1)

    if attn_mask is not None or (isinstance(topk, (int, float)) and topk == 1) or q.size(2) < 128:
        out = dense_attention(q, k, v, causal=is_causal, scale=scale, mask=attn_mask, dropout_p=dropout_p)
        sparsity = 0.0
    else:
        hopper = q.is_cuda and not torch.version.hip and torch.cuda.get_device_capability(q.device) == (9, 0)
        mq, mk = (64, 128) if hopper else (128, 64)
        qm, qs = _pool(q, mq)
        km, ks = _pool(k - k.mean(2, keepdim=True) if smooth_k else k, mk)
        if q.size(1) != k.size(1):
            groups = q.size(1) // k.size(1)
            km, ks = km.repeat_interleave(groups, 1), ks.repeat_interleave(groups, 1)
        score = (qm @ km.transpose(-1, -2)).float() * q.size(-1) ** -0.5
        valid = ks.unsqueeze(-2) > head_param(simthreshd1)
        validq = qs.unsqueeze(-1) > head_param(simthreshd1)
        nr, nk = score.shape[-2:]
        rows = torch.arange(nr, device=q.device)
        cols = torch.arange(nk, device=q.device)
        allowed = (
            cols[None, :] * mk < (rows[:, None] + 1) * mq
            if is_causal
            else torch.ones((nr, nk), device=q.device, dtype=torch.bool)
        )
        prob = score.masked_fill(~(valid & allowed), -torch.inf).softmax(-1).nan_to_num()
        ordered, indices = prob.sort(-1, descending=True)
        if topk is not None:
            count = (head_param(topk) * nk).long()
        else:
            count = (ordered.cumsum(-1) <= head_param(cdfthreshd)).sum(-1, keepdim=True)
        ranked = (cols < count.clamp(1, nk)).expand_as(indices)
        keep = torch.zeros_like(prob, dtype=torch.bool).scatter_(-1, indices, ranked)
        keep = (keep | ~valid | ~validq) & allowed
        if attention_sink:
            keep[..., 0] = True
        out = sparge_attention(
            q,
            k,
            v,
            keep,
            block_size=(mq, mk),
            scale=scale,
            causal=is_causal,
            dropout_p=dropout_p,
            smooth_k=smooth_k,
            pv_threshold=pvthreshd,
        )
        if return_sparsity:
            sparsity = 1.0 - keep.sum().item() / (q.size(0) * q.size(1) * allowed.sum().item())
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return (out, sparsity) if return_sparsity else out


def spas_sage2_attn_meansim_cuda(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    pvthreshd=50,
    attention_sink=False,
    tensor_layout="HND",
    output_dtype=torch.float16,
    return_sparsity=False,
):
    return _meansim(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        smooth_k,
        simthreshd1,
        cdfthreshd,
        None,
        attention_sink,
        pvthreshd,
        tensor_layout,
        return_sparsity,
    )


def spas_sage2_attn_meansim_topk_cuda(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    smooth_k=True,
    simthreshd1=-0.1,
    cdfthreshd=None,
    topk=0.5,
    pvthreshd=50,
    attention_sink=False,
    tensor_layout="HND",
    output_dtype=torch.float16,
    return_sparsity=False,
):
    return _meansim(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        smooth_k,
        simthreshd1,
        cdfthreshd,
        topk,
        attention_sink,
        pvthreshd,
        tensor_layout,
        return_sparsity,
    )


spas_sage_attn_meansim_cuda = spas_sage2_attn_meansim_cuda
spas_sage_attn_meansim_topk_cuda = spas_sage2_attn_meansim_topk_cuda
