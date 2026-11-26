"""Load platform-independent, precompiled device images through the CUDA or HIP driver."""

import ctypes as C
import functools
import os
from pathlib import Path
import torch

_IMAGES = Path(__file__).with_name("images")


def target(device):
    if torch.version.hip:
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def image_path(device):
    name = target(device)
    return _IMAGES / (name + (".hsaco" if torch.version.hip else ".cubin"))


class ImageUnavailable(RuntimeError):
    """This driver cannot load an image compiled for the requested architecture."""


class Module:
    def __init__(self, device):
        hip = bool(torch.version.hip)
        self.wave = 64 if hip and target(device).startswith("gfx9") else 32
        if hip:
            names = (
                ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll")
                if os.name == "nt"
                else ("libamdhip64.so", "libamdhip64.so.7")
            )
        else:
            names = ("nvcuda.dll",) if os.name == "nt" else ("libcuda.so.1",)
        for name in names:
            try:
                self.driver = C.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise OSError(f"Cannot load GPU driver {names}")
        prefix = "hip" if hip else "cu"
        self.launch_kernel = getattr(self.driver, prefix + ("ModuleLaunchKernel" if hip else "LaunchKernel"))
        self.functions = {}
        handle = C.c_void_p()
        self.image = image_path(device).read_bytes()
        # Discovery can run before PyTorch's first allocation. A CUDA device
        # guard alone may defer creation of its primary context on that thread.
        current = C.c_void_p()
        self.check(getattr(self.driver, prefix + "CtxGetCurrent")(C.byref(current)))
        primary = C.c_void_p()
        if not current:
            self.check(
                getattr(self.driver, prefix + "DevicePrimaryCtxRetain")(
                    C.byref(primary), device.index if device.index is not None else torch.cuda.current_device()
                )
            )
            push = getattr(self.driver, "hipCtxPushCurrent" if hip else "cuCtxPushCurrent_v2")
            self.check(push(primary))
        try:
            status = getattr(self.driver, prefix + "ModuleLoadData")(C.byref(handle), C.c_char_p(self.image))
            if not status:
                for name in ("sg_pack", "sg_select", "sg_attn64", "sg_attn128"):
                    function = C.c_void_p()
                    self.check(
                        getattr(self.driver, prefix + "ModuleGetFunction")(C.byref(function), handle, name.encode())
                    )
                    self.functions[name] = function
        finally:
            if primary:
                pop = getattr(self.driver, "hipCtxPopCurrent" if hip else "cuCtxPopCurrent_v2")
                self.check(pop(C.byref(current)))
        if status in (200, 209, 218, 222, 303):
            raise ImageUnavailable(f"The driver cannot load {image_path(device).name} ({status})")
        self.check(status)
        self.handle = handle

    def check(self, status):
        if status:
            raise RuntimeError(f"SparseGP GPU driver returned {status}")

    def launch(self, name, grid, args, device):
        values = [
            C.c_void_p(x.data_ptr())
            if isinstance(x, torch.Tensor)
            else C.c_float(x)
            if isinstance(x, float)
            else C.c_int(x)
            for x in args
        ]
        params = (C.c_void_p * len(values))(*(C.addressof(x) for x in values))
        stream = C.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        self.check(self.launch_kernel(self.functions[name], *grid, self.wave, 1, 1, 0, stream, params, None))


@functools.lru_cache(None)
def module(device):
    with torch.cuda.device(device):
        return Module(device)


def pack(x):
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    module(x.device).launch("sg_pack", (scales.numel(), 1, 1), (x, out, scales, x.shape[-1]), x.device)
    return out, scales


def attend(q, k, v, qs, ks, bits, keys, block_q, scale, representatives, causal):
    groups, qp, width = q.shape
    out = torch.empty_like(q, dtype=torch.float16)
    module(q.device).launch(
        f"sg_attn{width}",
        (qp // 16, groups, 1),
        (q, k, v, qs, ks, bits, out, qp, k.size(1), keys, block_q, float(scale), int(representatives), int(causal)),
        q.device,
    )
    return out


def select(qmean, keys, flags, tokens, scale):
    from .routing import pad, round_up

    g, n, d = qmean.shape
    padded = pad(qmean, round_up(n, 16))
    stats = torch.empty((g, n, n, 3), device=keys.device, dtype=torch.float32)
    bits = torch.empty((g, n, n, 2), device=keys.device, dtype=torch.int32)
    grid = ((n + 15) // 16, n, g)
    launch = module(keys.device).launch

    def run(cut, phase):
        launch(
            "sg_select",
            grid,
            (padded, keys, flags, stats, cut, bits, n, padded.size(1), tokens, d, float(scale), phase),
            keys.device,
        )

    run(stats, 0)
    count, total, squared = stats.sum(2).unbind(-1)
    count.clamp_min_(1.0)
    mean = total / count
    cut = mean + 1.125 * (torch.clamp_min(squared / count - mean.square(), 0.0) + 1.0e-6).sqrt()
    run(cut, 1)
    return bits
