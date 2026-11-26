"""Build shape-independent GPU binaries for mixed attention and projection packing.

Triton is a build dependency. Token counts, strides, scale and block-map contents are runtime arguments.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import triton
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

knobs.compilation.disable_line_info = True

root = Path(__file__).resolve().parents[1] / "sparsegp"


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(arch):
    hip = arch.startswith("gfx")
    target = GPUTarget("hip", arch, 64 if arch.startswith("gfx9") else 32) if hip else GPUTarget("cuda", int(arch), 32)
    folder = root / "images" / (arch if hip else "sm_" + arch)
    folder.mkdir(exist_ok=True, parents=True)
    binary = "hsaco" if hip else "cubin"
    simt = arch == "75" or arch.startswith("gfx10")
    entries = {}

    def emit(name, fn, arguments, constants, stages=1):
        dynamic = [k for k in fn.arg_names if k not in constants]
        sig = dict(zip(dynamic, arguments.split(), strict=True))
        sig = {k: sig.get(k, "constexpr") for k in fn.arg_names}
        compiled = triton.compile(
            ASTSource(fn, sig, constants),
            target=target,
            options={"num_warps": 4, "num_stages": stages, "enable_fp_fusion": False},
        )
        assert not getattr(compiled.metadata, "global_scratch_size", 0)
        (folder / f"{name}.{binary}").write_bytes(compiled.asm[binary])
        entries[name] = [compiled.metadata.name, compiled.metadata.shared, 4 * target.warp_size]

    for d in (64, 128):
        emit(f"pack{d}", mixed._pack, "*fp16 *i8 *fp32", dict(D=d, ROWS=16))
        for reps, causal in ((False, False), (False, True), (True, False), (True, True)):
            constants = dict(D=d, CAUSAL=causal, REPS=reps, WHOLE=False, Counts=None)
            if simt:
                constants["HIP"] = hip
            else:
                constants["M"] = 32 if hip or d == 128 else 64
            emit(
                f"attend{d}_{int(reps)}{int(causal)}",
                mixed._attend_dp4a if simt else mixed._attend,
                "*i8 *i8 *fp16 *fp32 *fp32 *i32 *fp16 i32 i32 i32 i32 fp32",
                constants,
            )
        for causal in (False, True):
            constants = dict(D=d, CAUSAL=causal, REPS=False, WHOLE=True)
            constants.update(dict(HIP=hip) if simt else dict(M=32 if hip or d == 128 else 64))
            emit(
                f"sparge{d}_{int(causal)}",
                mixed._attend_dp4a if simt else mixed._attend,
                "*i8 *i8 *fp16 *fp32 *fp32 *i32 *fp16 i32 i32 i32 i32 fp32 *i32",
                constants,
            )
        for write in (False, True):
            emit(
                f"select{d}_{int(write)}",
                mixed._select_simt if simt else mixed._select,
                "*fp16 *fp16 *i8 *fp32 *fp32 *i32 i32 i32 fp32",
                dict(D=d, WRITE=write),
            )
        if not hip and int(arch) >= 80:
            for dtype in ("fp16", "bf16"):
                formats = [("q", 128, False), ("k", 64, True), ("kraw", 64, False)]
                if arch == "90":
                    formats += [("q64", 64, False), ("k128", 128, True), ("kraw128", 128, False)]
                for kind, block, sub in formats:
                    emit(
                        f"{kind}_{d}_{dtype}",
                        prep._quant_block_nhd,
                        f"*{dtype} *{dtype} *i8 *fp32 " + " ".join(["i32"] * 10),
                        dict(D=d, BLK=block, SUB_MEAN=sub),
                    )
                if int(arch) >= 89:
                    emit(
                        f"v_{d}_{dtype}",
                        prep._v_fp8_nhd,
                        f"*{dtype} *{dtype} *{dtype} *fp8e4nv *fp32 i32 i32 i32 i32 i32 i32 fp32",
                        dict(D=d, BLOCK=64),
                    )
    for causal in (False, True):
        for delta in (False, True) if not hip and int(arch) >= 80 else (False,):
            name = "lut" if delta else "indices"
            emit(
                f"{name}{int(causal)}",
                prep._lookup,
                "*i8 *i32 *i32 i32 i32 i32 i32 i32 i32 i32 i32 i32",
                dict(CAUSAL=causal, DELTA=delta, CHUNK=256),
            )
    (folder / "kernels.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(arch, len(entries), "compiled kernels", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args()
    mixed = source("sparsegp_mixed_build", root / "triton_backend.py")
    prep = source("sparsegp_prepare_build", root / "csrc/prepare.py")
    for arch in args.targets:
        build(arch)
