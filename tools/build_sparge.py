"""Compile SpargeAttention's upstream tensor-core code into driver-loadable CUDA binaries."""

import argparse
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("targets", nargs="+", type=int)
parser.add_argument("--nvcc", default="nvcc")
parser.add_argument("--ccbin")
args = parser.parse_args()
for target in args.targets:
    fp8 = target >= 89
    subprocess.run(
        [
            args.nvcc,
            "--cubin",
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            *(["-ccbin", args.ccbin] if args.ccbin else []),
            f"-arch=sm_{target}",
            f"-DSPARGE_FP8={int(fp8)}",
            f"-DSPARGE_HALF_PV={int(target >= 120)}",
            str(root / "sparsegp/csrc/sparge.cu"),
            "-o",
            str(root / f"sparsegp/images/sparge_sm_{target}.cubin"),
        ],
        check=True,
    )

    if target == 90:
        subprocess.run(
            [
                args.nvcc,
                "--cubin",
                "-std=c++17",
                "-O3",
                "--use_fast_math",
                *(["-ccbin", args.ccbin] if args.ccbin else []),
                "-arch=sm_90a",
                str(root / "sparsegp/csrc/sparge_hopper.cu"),
                "-o",
                str(root / "sparsegp/images/sparge_sm_90a.cubin"),
            ],
            check=True,
        )
