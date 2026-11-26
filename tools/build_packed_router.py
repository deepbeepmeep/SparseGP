"""Build the packed BF16 router with Triton 3.6. packed_cuda/build.py assembles these build intermediates into the release cubins."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

folder = Path(__file__).resolve().parents[1] / "sparsegp" / "packed_cuda"
spec = importlib.util.spec_from_file_location("router_kernels", folder / "router_kernels.py")
kernels = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernels
spec.loader.exec_module(kernels)

parser = argparse.ArgumentParser()
parser.add_argument("architectures", nargs="+", type=int, choices=[89, 120])
args = parser.parse_args()
for arch in args.architectures:
    shapes = {}

    def build(name, fn, arguments, constants, stages=3):
        signature = dict(zip(fn.arg_names, arguments.split(), strict=False))
        signature.update({k: "constexpr" for k in constants})
        assert set(signature) == set(fn.arg_names)
        # Allocated tensors and the packed input rows are 16-byte aligned.
        attrs = {(i,): [["tt.divisibility", 16]] for i, (arg, typ) in enumerate(signature.items())
                 if typ.startswith("*") or arg in ("ST", "SH")}
        code = triton.compile(
            ASTSource(fn, signature, constants, attrs),
            target=GPUTarget("cuda", arch, 32),
            options={"num_warps": 4, "num_stages": stages},
        )
        ptx = code.asm["ptx"]
        # Keep source line information without embedding the build machine's paths.
        ptx = re.sub(
            r'(?m)^(\s*\.file\s+\d+\s+)"([^"]+)"',
            lambda m: m[1] + '"' + m[2].replace(chr(92), "/").rsplit("/", 1)[-1] + '"',
            ptx,
        )
        entry = code.metadata.name
        params = re.search(r"\.entry\s+\w+\s*\((.*?)\)", ptx, re.S).group(1).count(".param")
        extra = params - len(arguments.split())
        # Triton adds unused global and profiling scratch pointers to the launch ABI.
        assert extra in (0, 1, 2) and not getattr(code.metadata, "global_scratch_size", 0)
        (folder / "ptx" / f"route_{arch}_{name}.ptx").write_text(ptx, newline="\n")
        shapes[name] = [entry, code.metadata.shared, extra]

    for kind in ("query", "key", "value"):
        build(kind, kernels._pool, "*bf16 *fp32 i32 i32 i32 i32", {"KIND": kind})
    # The torch CUDA image launcher passes Python floats as doubles. The kernels cast to FP32 at entry.
    build(
        "classify",
        kernels._classify,
        "*fp32 *bf16 *fp32 *fp32 *fp32 *fp32 *fp32 *fp32 *i8 i32 fp64 fp64 i32 i32 i32",
        {"BQ": 16, "BK": 64},
    )
    for phase, name in ((False, "moments"), (True, "select")):
        build(
            name,
            kernels._tokens,
            "*fp32 *bf16 *i8 *fp32 *fp32 *i64 i32 i32 i32 i32 i32 fp64 i32",
            {"SELECT": phase, "BQ": 32, "BK": 256},
            stages=2,
        )
    build("cut", kernels._cut, "*fp32 *fp32 i32", {})
    (folder / "ptx" / f"route_{arch}.json").write_text(json.dumps(shapes, indent=2) + "\n", newline="\n")
