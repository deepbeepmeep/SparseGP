# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Sol³ mixed-block routing for WanGP's quantized attention (FofGofx/sol3, SM120 router).
All routing buffers belong to the call. Only the original keys survive staging, for scoring individual tokens.
"""
import triton
import triton.language as tl


@triton.jit
def _pool(X, OUT, T, N, ST, SH, KIND: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    values = tl.load(X + rows[:, None].to(tl.int64) * ST + head * SH + cols[None, :], rows[:, None] < T, 0).to(tl.float32)
    length = tl.minimum(64, T - block * 64)
    mean = tl.sum(values, 0) / length
    dest = OUT + (head * N + block) * 128 + cols
    if KIND == "query":
        tl.store(dest, mean)
    else:
        delta = tl.where(rows[:, None] < T, values - mean[None, :], 0.)
        var = tl.sum(delta * delta, 0) / length
        if KIND == "key":
            tl.store(dest, var)
        else:
            rms = tl.sqrt(tl.sum(var, 0) / 128.)
            tl.store(OUT + head * N + block, tl.log2(length * tl.maximum(rms, 1.e-6)))


@triton.jit
def _classify(Q, KC, VAR, DISP, MEAN, BETWEEN, AVG_VAR, AVG_DISP, FLAGS,
              N, SCALE, TAU, SINK0, SINK1, BQ: tl.constexpr, BK: tl.constexpr):
    SCALE = SCALE.to(tl.float32)
    TAU = TAU.to(tl.float32)
    qi = tl.program_id(0) * BQ + tl.arange(0, BQ)
    ki = tl.program_id(1) * BK + tl.arange(0, BK)
    head = tl.program_id(2)
    d = tl.arange(0, 128)
    q = tl.load(Q + (head * N + qi[:, None]) * 128 + d[None, :], qi[:, None] < N, 0)
    k = tl.load(KC + (head * N + ki[:, None]) * 128 + d[None, :], ki[:, None] < N, 0).to(tl.float32)
    kvar = tl.load(VAR + (head * N + ki[:, None]) * 128 + d[None, :], ki[:, None] < N, 0)
    disp = tl.load(DISP + head * N + ki, ki < N, 0)
    s = SCALE * 1.4426950408889634
    proxy = tl.dot(q, tl.trans(k), input_precision="tf32x3") * s
    q2 = q * q
    mu = tl.sum(q * tl.load(MEAN + head * 128 + d)[None, :], 1) * s
    var = tl.sum(q2 * tl.load(BETWEEN + head * 128 + d)[None, :], 1) * (s * s)
    full = (proxy > (mu + TAU * tl.sqrt(var + 1.e-6))[:, None]) | (tl.abs(qi[:, None] - ki[None, :]) <= 1)
    full |= ((ki >= SINK0) & (ki < SINK1))[None, :]
    fac = 0.5 * SCALE * SCALE * 1.4426950408889634
    risk = proxy + disp[None, :] + fac * tl.dot(q2, tl.trans(kvar), input_precision="tf32x3")
    # For beta2=0 and uncapped diagonal variance, the row's mean risk is linear in the block statistics.
    cut = mu + tl.load(AVG_DISP + head) + fac * tl.sum(q2 * tl.load(AVG_VAR + head * 128 + d)[None, :], 1)
    flags = tl.where(full, -1, (risk >= cut[:, None]).to(tl.int32))
    tl.store(FLAGS + (head * N + qi[:, None]) * N + ki[None, :], flags, (qi[:, None] < N) & (ki[None, :] < N))


@triton.jit
def _tokens(Q, K, FLAGS, PARTIAL, CUT, BITS, T, N, NT, ST, SH, SCALE,
            SELECT: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr):
    SCALE = SCALE.to(tl.float32)
    qi = tl.program_id(0) * BQ + tl.arange(0, BQ)
    tile, head = tl.program_id(1), tl.program_id(2)
    blocks = tile * (BK // 64) + tl.arange(0, BK // 64)
    row = head * N + qi
    valid = (qi[:, None] < N) & (blocks[None, :] < N)
    flags = tl.load(FLAGS + row[:, None] * N + blocks[None, :], valid, 0)
    keys = tile * BK + tl.arange(0, BK)
    mixed = tl.broadcast_to((flags == 1)[:, :, None], (BQ, BK // 64, 64))
    eligible = tl.reshape(mixed, (BQ, BK)) & (keys < T)[None, :]
    score = tl.full((BQ, BK), 0, tl.float32)
    if tl.sum(tl.sum((flags == 1).to(tl.int32), 1), 0) > 0:
        d = tl.arange(0, 128)
        q = tl.load(Q + row[:, None] * 128 + d[None, :], qi[:, None] < N, 0).to(tl.bfloat16)
        k = tl.load(K + head * SH + keys[:, None].to(tl.int64) * ST + d[None, :], keys[:, None] < T, 0)
        score = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
    if SELECT:
        cut = tl.load(CUT + row, qi < N, float("inf"))
        chosen = tl.reshape(eligible & (score > cut[:, None]), (BQ, BK // 64, 64))
        bit = tl.full((64,), 1, tl.uint64) << tl.arange(0, 64).to(tl.uint64)
        bits = tl.sum(tl.where(chosen, bit[None, None, :], 0), 2)
        tl.store(BITS + row[:, None] * N + blocks[None, :], tl.where(flags == -1, ~tl.full((), 0, tl.uint64), bits), valid)
    else:
        count = tl.sum(eligible.to(tl.float32), 1)
        mean = tl.sum(tl.where(eligible, score, 0.), 1) / tl.maximum(count, 1.)
        diff = tl.where(eligible, score - mean[:, None], 0.)
        base = PARTIAL + (row * NT + tile) * 3
        tl.store(base, count, qi < N)
        tl.store(base + 1, mean, qi < N)
        tl.store(base + 2, tl.sum(diff * diff, 1), qi < N)


@triton.jit
def _cut(PARTIAL, CUT, NT):
    row = tl.program_id(0)
    lanes = tl.arange(0, 128)
    base = PARTIAL + row * NT * 3
    counts = tl.full((128,), 0, tl.float32)
    means = tl.full((128,), 0, tl.float32)
    for start in range(0, NT, 128):
        i = start + lanes
        n = tl.load(base + i * 3, i < NT, 0.)
        counts += n
        means += n * tl.load(base + i * 3 + 1, i < NT, 0.)
    count = tl.maximum(tl.sum(counts, 0), 1.)
    mu = tl.sum(means, 0) / count
    squared = tl.full((128,), 0, tl.float32)
    for start in range(0, NT, 128):
        i = start + lanes
        n = tl.load(base + i * 3, i < NT, 0.)
        delta = tl.load(base + i * 3 + 1, i < NT, 0.) - mu
        squared += tl.load(base + i * 3 + 2, i < NT, 0.) + n * delta * delta
    tl.store(CUT + row, mu + 1.125 * tl.sqrt(tl.sum(squared, 0) / count + 1.e-6))
