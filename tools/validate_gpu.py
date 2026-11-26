"""Run on the target GPU after installing the wheel. Fails on a different target or backend."""

import argparse
import os
import statistics
import torch
from torch.nn import functional as F

from sparsegp import sparge_cuda, backend_for, dense_attention, sol3_attention, sparge_attention
from sparsegp.core import _backend
from sparsegp.native import target
from sparsegp.routing import sol_route
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda


@torch.inference_mode()
def validate(device, backend):
    validate_grouped(device, backend)
    torch.manual_seed(132)
    d = 64
    tokens = 1089
    q, k, v = [torch.randn(1, tokens, 2, d, device=device, dtype=torch.float16) for _ in range(3)]
    # Repeated neighborhoods and dispersed values exercise the three routes.
    q[:, 128:192].mul_(3)
    k[:, 448:512].add_(1)
    v[:, 640:704].mul_(4)
    x, y, z = [t.transpose(1, 2).squeeze(0).contiguous() for t in (q, k, v)]
    engine = _backend(device, backend)
    scale = d**-0.5
    bits, kc, vc = sol_route(x, y, z, scale, 1.9, 0, 37, engine)
    n = kc.size(1)
    lanes = torch.arange(32, device=device)
    chosen = (((bits[..., None] >> lanes) & 1) > 0).flatten(-2)
    length = (tokens - 64 * torch.arange(n, device=device)).clamp_max(64)
    count = chosen.sum(-1)
    mixed = (count > 0) & (count < length)
    assert mixed.any() and (count == 0).any() and (count == length).any()
    rows = torch.arange(38, tokens, 17, device=device)
    keep = chosen[:, rows // 64].flatten(-2)[..., :tokens]
    mass = length - count[:, rows // 64]
    exact = (x[:, rows].float() @ y.float().transpose(1, 2)) * scale
    centroid = (x[:, rows].float() @ kc.float().transpose(1, 2)) * scale + mass.float().log()
    p = torch.cat((exact.masked_fill(~keep, -torch.inf), centroid), -1).softmax(-1)
    expected = p @ torch.cat((z, vc), 1).float()
    actual = sol3_attention(q, k, v, prefix_tokens=37, backend=backend)
    got = actual[0, rows].transpose(0, 1).float()
    rel = (got - expected).norm() / expected.norm()
    assert rel < 0.025, (backend, "representative mass", float(rel))
    mask = torch.rand(1, 2, 2, 4, device=device) > 0.35
    mask[:, :, 0] = False
    a, b, c = [
        t[:, :, :n].contiguous()
        for t, n in zip((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), (129, 193, 193))
    ]
    o = sparge_attention(a, b, c, mask, backend=backend, causal=True)
    qm = torch.arange(129, device=device)[:, None]
    km = torch.arange(193, device=device)[None, :]
    visible = mask[:, :, qm // 128, km // 64] & (km <= qm)
    reference = F.scaled_dot_product_attention(a.float(), b.float(), c.float(), attn_mask=visible)
    block_error = (o - reference).norm() / reference.norm().clamp_min(1.0e-10)
    # Upstream FP8 weights both the values and the softmax probabilities (about 4% RMS on random inputs).
    fp8_sparge = backend == "cuda" and bool((sparge_cuda.settings(device, d) or {}).get("fp8"))
    assert torch.isfinite(o).all() and block_error < (0.055 if fp8_sparge else 0.025), (backend, float(block_error))
    assert not torch.count_nonzero(o[~visible.any(-1)])

    # Replay with changed projections and a different set of visible blocks.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_sparse = sparge_attention(a, b, c, mask, backend=backend, causal=True)
    a.neg_()
    mask.logical_not_()
    expected_sparse = sparge_attention(a, b, c, mask, backend=backend, causal=True)
    graph.replay()
    torch.testing.assert_close(captured_sparse, expected_sparse, rtol=0, atol=0)

    dense = spas_sage2_attn_meansim_topk_cuda(a, b, c, topk=1.0)
    torch.testing.assert_close(dense, dense_attention(a, b, c))
    if backend == backend_for(device):
        from sparsegp import staged

        an, bn, cn = (x.transpose(1, 2) for x in (a, b, c))
        settings = staged.staged_settings(device, d)
        projected = [
            staged.staged_quantize_q(an, settings),
            staged.staged_quantize_k([bn], settings),
            staged.staged_quantize_v([cn], settings),
        ]
        staged_out = staged.staged_attention(torch.empty_like(an), projected, settings, mask)
        torch.testing.assert_close(staged_out.transpose(1, 2), sparge_attention(a, b, c, mask))
    # Replay after input changes; static-pointer bugs pass an unchanged-input check.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sol3_attention(q, k, v, backend=backend)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sol3_attention(q, k, v, backend=backend)
    q.mul_(-0.8)
    expected = sol3_attention(q, k, v, backend=backend)
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0.0002, rtol=0.001)
    samples = []
    for i in range(7):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        sol3_attention(q, k, v, backend=backend)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    print(
        f"{target(device)} {backend}: Sol3 relative RMS {float(rel):.5f}, Sparge relative RMS {float(block_error):.5f}, "
        f"graph replay passed, {tokens} tokens / 2 heads / width 64 median {statistics.median(samples):.3f} ms"
    )


@torch.inference_mode()
def validate_grouped(device, backend):
    torch.manual_seed(672)
    for width in (64, 128):
        q = torch.randn(1, 1089, 8, width, dtype=torch.bfloat16, device=device)
        k, v = [torch.randn_like(q[:, :, :2]) for _ in range(2)]
        options = dict(tau=0.8, prefix_tokens=17, backend=backend)
        expected = sol3_attention(q, k.repeat_interleave(4, 2), v.repeat_interleave(4, 2), **options)
        actual = sol3_attention(q, k, v, **options)
        torch.testing.assert_close(actual[:, 17:], expected[:, 17:], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, :17], expected[:, :17], atol=.007, rtol=.03)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            recorded = sol3_attention(q, k, v, **options)
        k.neg_()
        graph.replay()
        torch.testing.assert_close(recorded, sol3_attention(q, k, v, **options), atol=0, rtol=0)
    print(f"{backend} grouped-query routing and graph replay passed")


@torch.inference_mode()
def validate_packed(device):
    from sparsegp import packed_cuda

    if not packed_cuda.supported(device):
        return
    torch.manual_seed(572)
    q, k, v = [torch.randn((1, 1153, 2, 128), device=device, dtype=torch.bfloat16) for _ in range(3)]
    from sparsegp.packed_cuda import packed_routing

    prepared = packed_cuda.prepare(k, v, mixed=True)
    score_scale = q.size(-1) ** -0.5
    got = packed_routing.select(q, *prepared[6], prepared[3], prepared[4], score_scale, 0.8, (0, 1))
    x, y, z = (t[0].transpose(0, 1).half().contiguous() for t in (q, k, v))
    expected_bits, _, _ = sol_route(x, y, z, score_scale, 0.8, 0, 37, _backend(device, "cuda"))
    lanes = torch.arange(32, device=device)
    got = got.contiguous().view(torch.int32).reshape_as(expected_bits)
    disagreement = (((got ^ expected_bits).unsqueeze(-1) >> lanes) & 1).float().mean()
    assert disagreement < 0.025, ("packed routing bit disagreement", float(disagreement))
    print(f"{target(device)} packed routing bit disagreement against the compiled mixed-block reference: {float(disagreement):.5f}")
    expected = F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in (q, k, v))).transpose(1, 2)
    for precision in ("int8", "fp4") if target(device) == "sm_120" else ("int8",):
        full = sol3_attention(q, k, v, tau=-1.0e6, precision=precision).float()
        error = (full - expected).norm() / expected.norm()
        assert torch.isfinite(full).all() and error < (0.1 if precision == "int8" else 0.45), (precision, error)
        shared_k, shared_v = k[:, :, :1], v[:, :, :1]
        grouped = sol3_attention(q, shared_k, shared_v, precision=precision)
        repeated = sol3_attention(q, shared_k.repeat_interleave(2, 2), shared_v.repeat_interleave(2, 2), precision=precision)
        torch.testing.assert_close(grouped, repeated, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = sol3_attention(q, k, v, precision=precision)
        k.neg_()
        eager = sol3_attention(q, k, v, precision=precision)
        graph.replay()
        torch.testing.assert_close(result, eager, atol=0.001, rtol=0.002)
        k.neg_()
        print(
            f"{target(device)} packed {precision}: full-route relative RMS {float(error):.5f}, mixed graph replay passed"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=os.environ.get("SPARSEGP_TARGET"), help="Require this specific device architecture")
    parser.add_argument("--triton", action="store_true", help="Exercise the JIT fallback as well")
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    if args.target and target(device) != args.target:
        raise SystemExit(f"Expected {args.target}, found {target(device)}")
    name = backend_for(device)
    if not name:
        raise SystemExit("No supported sparse kernel")
    validate(device, name)
    if name == "cuda":
        validate_packed(device)
    if args.triton and name != "triton":
        from unittest.mock import patch
        from sparsegp import native
        from sparsegp.core import _device_backend

        # Model a driver rejecting an otherwise installed image and exercise automatic selection.
        try:
            _device_backend.cache_clear()
            with patch.object(native, "module", side_effect=native.ImageUnavailable("unsupported test image")):
                assert backend_for(device) == "triton"
                with torch.inference_mode():
                    x = torch.randn(1, 1025, 1, 64, device=device, dtype=torch.float16)
                    torch.testing.assert_close(sol3_attention(x, x, x), sol3_attention(x, x, x, backend="triton"))
        finally:
            _device_backend.cache_clear()
        validate(device, "triton")
