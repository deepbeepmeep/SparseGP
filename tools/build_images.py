"""Build isolated device images; no Python, operating-system or libtorch ABI is embedded."""

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1] / "sparsegp"
parser = argparse.ArgumentParser()
parser.add_argument("targets", nargs="+")
parser.add_argument("--nvcc", default="nvcc")
parser.add_argument("--rocm", type=Path)
args = parser.parse_args()
out = ROOT / "images"
out.mkdir(exist_ok=True)
for arch in args.targets:
    source = str(ROOT / "csrc/attention.cu")
    if arch.startswith("gfx"):
        sdk = args.rocm or Path(os.environ["ROCM_PATH"])
        exe = sdk / ("lib/llvm/bin/clang++.exe" if os.name == "nt" else "lib/llvm/bin/clang++")
        cmd = [
            str(exe),
            "-x",
            "hip",
            "--offload-device-only",
            "--offload-arch=" + arch,
            "--rocm-path=" + str(sdk),
            "--rocm-device-lib-path=" + str(sdk / "lib/llvm/amdgcn/bitcode"),
            "-I" + str(sdk / "include"),
            "-std=c++17",
            "-O3",
            source,
            "-o",
            str(out / (arch + ".hsaco")),
        ]
    else:
        cmd = [
            args.nvcc,
            "--cubin",
            "-O3",
            "-std=c++17",
            "-arch=" + arch,
            source,
            "-o",
            str(out / (arch + ".cubin")),
        ]
    subprocess.run(cmd, check=True)
    if arch.startswith("gfx"):
        image = out / (arch + ".hsaco")
        if not image.read_bytes().startswith(b"\x7fELF"):
            bundle = image.with_suffix(".bundle")
            image.replace(bundle)
            bundler = exe.with_name("clang-offload-bundler" + exe.suffix)
            subprocess.run(
                [
                    str(bundler),
                    "--unbundle",
                    "--type=o",
                    "--targets=hipv4-amdgcn-amd-amdhsa--" + arch,
                    "--input=" + str(bundle),
                    "--output=" + str(image),
                ],
                check=True,
            )
            bundle.unlink()
