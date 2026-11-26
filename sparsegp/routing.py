"""Sol³ block relevance, approximation risk and mixed-token selection."""

import math
import torch
from torch.nn import functional as F


def round_up(length, tile):
    return (length + tile - 1) // tile * tile


def pad(x, size):
    return x.contiguous() if size == x.size(1) else F.pad(x, (0, 0, 0, size - x.size(1))).contiguous()


def moments(x, n):
    groups, tokens, dim = x.shape
    rows = pad(x, n * 64).view(groups, n, 64, dim)
    length = (tokens - torch.arange(n, device=x.device) * 64).clamp(max=64).float()
    var, mean = torch.var_mean(rows.float(), dim=2, correction=0)
    # var_mean on FP32 tiles preserves the routing of nearly constant blocks.
    if tokens % 64:
        tail_var, tail_mean = torch.var_mean(rows[:, -1, : tokens % 64].float(), dim=1, correction=0)
        var[:, -1], mean[:, -1] = tail_var.to(var.dtype), tail_mean.to(mean.dtype)
    return mean.to(x.dtype), var, length


def prepare_sol_kv(k, v):
    n = (k.size(1) + 63) // 64
    km, kvar, lengths = moments(k, n)
    vm, vvar, _ = moments(v, n)
    disp = (lengths * vvar.float().mean(-1).clamp_min(0).sqrt().clamp_min(1.e-6)).log2()
    return km, vm, kvar, disp


def route_queries(q, k, km, kvar, disp, scale, tau, sink_start, sink_length, engine):
    tokens = q.size(1)
    n = (tokens + 63) // 64
    qm, _, _ = moments(q, n)
    s = scale * math.log2(math.e)
    qf, kf, kvar = qm.float(), km.float(), kvar.float()
    between, center = torch.var_mean(kf, dim=1, correction=0, keepdim=True)
    mu = (qf * center).sum(-1) * s
    sigma = ((qf.square() * between).sum(-1) * s * s + 1.0e-6).sqrt()
    idx = torch.arange(n, device=q.device)
    neighbor = (idx[:, None] - idx[None, :]).abs() <= 1
    if sink_length:
        neighbor |= ((idx * 64 < sink_start + sink_length) & ((idx + 1) * 64 > sink_start))[None, :]
    proxy = torch.bmm(qm, km.transpose(1, 2)).float() * s
    q2 = qf.square()
    fac = 0.5 * scale * scale * math.log2(math.e)
    risk = proxy + disp[:, None, :] + fac * torch.bmm(q2, kvar.transpose(1, 2))
    cut = mu + disp.mean(-1, keepdim=True) + fac * (q2 * kvar.mean(1, keepdim=True)).sum(-1)
    full = (proxy > (mu + tau * sigma)[:, :, None]) | neighbor
    flags = torch.where(full, -1, (risk >= cut[:, :, None]).to(torch.int8)).to(torch.int8)
    bits = engine.select(qm, pad(k, n * 64), flags, tokens, s)
    return bits


def sol_route(q, k, v, scale, tau, sink_start, sink_length, engine):
    km, vm, kvar, disp = prepare_sol_kv(k, v)
    return route_queries(q, k, km, kvar, disp, scale, tau, sink_start, sink_length, engine), km, vm
