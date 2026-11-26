# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Assemble packed Sol attention, packing and routing machine images at build time.

Pass --regenerate-ptx to rebuild the Sol CUDA intermediates with NVRTC before assembly.
The wheel contains architecture-specific cubins for direct driver loading.
"""
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# image: (source, architecture, defines)
IMAGES = {
    "attn_i8_sm89": ("sol_attn.cu", "89", ("KERNEL_NAME=sol_attn_i8", "QK_FP4=0", "BULK=0")),
    "attn_i8_mixed_sm89": ("sol_attn.cu", "89", ("KERNEL_NAME=sol_attn_i8_mixed", "QK_FP4=0", "MIXED_ROUTING=1", "BULK=0")),
    "attn_i8_sm120a": ("sol_attn.cu", "120a", ("KERNEL_NAME=sol_attn_i8", "QK_FP4=0", "BULK=1")),
    "attn_fp4_sm120a": ("sol_attn.cu", "120a", ("KERNEL_NAME=sol_attn_fp4", "QK_FP4=1", "BULK=1")),
    "attn_i8_mixed_sm120a": ("sol_attn.cu", "120a", ("KERNEL_NAME=sol_attn_i8_mixed", "QK_FP4=0", "MIXED_ROUTING=1")),
    "attn_fp4_mixed_sm120a": ("sol_attn.cu", "120a", ("KERNEL_NAME=sol_attn_fp4_mixed", "QK_FP4=1", "MIXED_ROUTING=1")),
    "pack_sm89": ("sol_pack.cu", "89", ("FP4=0",)),
    "pack_fp4_sm120a": ("sol_pack.cu", "120a", ("FP4=1",)),
}
ENTRY = {"attn_i8_mixed_sm89": "sol_attn_i8_mixed", "attn_i8_sm89": "sol_attn_i8", "attn_i8_sm120a": "sol_attn_i8", "attn_fp4_sm120a": "sol_attn_fp4", "pack_sm89": "sol_pack_v",
         "pack_fp4_sm120a": "sol_pack_k4", "attn_i8_mixed_sm120a": "sol_attn_i8_mixed", "attn_fp4_mixed_sm120a": "sol_attn_fp4_mixed"}


def main():
    for image, (source, arch, defines) in IMAGES.items():
        ptx, _ = torch.cuda._utils._nvrtc_compile((HERE / source).read_text(encoding="utf-8"), ENTRY[image], compute_capability=arch,
                                                 nvcc_options=[f"-D{d}" for d in defines])
        (HERE / "ptx" / f"{image}.ptx").write_bytes(ptx)
        print(image, len(ptx), "bytes")


if __name__ == "__main__":
    import argparse
    import shutil
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-ptx", action="store_true", help="Regenerate Sol PTX from the CUDA sources before assembly")
    parser.add_argument("--ptxas",default="ptxas")
    args = parser.parse_args()
    if args.regenerate_ptx:
        main()
    folder = HERE / "images"
    folder.mkdir(exist_ok=True)
    for source in sorted((HERE/"ptx").glob("*.ptx")):
        name = source.stem
        arch = "sm_120a" if "120" in name else "sm_89"
        for target in ((arch,"sm_120a") if name=="pack_sm89" else (arch,)):
            output = "pack_sm120a" if name=="pack_sm89" and target=="sm_120a" else name
            subprocess.run([args.ptxas, "-arch="+target, str(source), "-o", str(folder/(output+".cubin"))], check=True)
    for manifest in (HERE/"ptx").glob("route_*.json"):
        shutil.copyfile(manifest,folder/manifest.name)
