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
    return _IMAGES / target(device) / "kernels.json"


class ImageUnavailable(RuntimeError):
    """This driver cannot load an image compiled for the requested architecture."""


class Module:
    def __init__(self, device, path):
        hip = bool(torch.version.hip)
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
        self.image = path.read_bytes()
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
            self.get_function = getattr(self.driver, prefix + "ModuleGetFunction")
        finally:
            if primary:
                pop = getattr(self.driver, "hipCtxPopCurrent" if hip else "cuCtxPopCurrent_v2")
                self.check(pop(C.byref(current)))
        if status in (200, 209, 218, 222, 303):
            raise ImageUnavailable(f"The driver cannot load {path.name} ({status})")
        self.check(status)
        self.handle = handle

    def check(self, status):
        if status:
            raise RuntimeError(f"SparseGP GPU driver returned {status}")

    def function(self, name):
        if name not in self.functions:
            func = C.c_void_p()
            self.check(self.get_function(C.byref(func), self.handle, name.encode()))
            self.functions[name] = func
        return self.functions[name]

    def launch(self, name, grid, block, args, device, shared=0):
        values = [
            C.c_void_p(x.data_ptr())
            if isinstance(x, torch.Tensor)
            else C.c_float(x)
            if isinstance(x, float)
            else C.c_int(x)
            if isinstance(x, int)
            else x
            for x in args
        ]
        params = (C.c_void_p * len(values))(*(C.addressof(x) for x in values))
        stream = C.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        fn = self.function(name)
        if shared >= 49152 and not torch.version.hip:
            self.check(self.driver.cuFuncSetAttribute(fn, 8, shared))
        self.check(self.launch_kernel(fn, *grid, *block, shared, stream, params, None))


@functools.lru_cache(None)
def load_image(device, path):
    with torch.cuda.device(device):
        return Module(device, path)


@functools.lru_cache(None)
def catalog(device):
    import json

    return json.loads(image_path(device).read_text())


@functools.lru_cache(None)
def entry(device, name):
    symbol, shared, threads = catalog(device)[name]
    ext = ".hsaco" if torch.version.hip else ".cubin"
    image = load_image(device, image_path(device).parent / (name + ext))
    image.function(symbol)
    return image, symbol, shared, threads


def module(device):
    return entry(device, "pack64")[0]


def launch(device, name, grid, args):
    image, symbol, shared, threads = entry(device, name)
    # Current Triton AOT ABIs append unused global/profiling scratch pointers.
    image.launch(symbol, grid, (threads, 1, 1), (*args, C.c_void_p(), C.c_void_p()), device, shared)


def pack(x):
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    launch(x.device, f"pack{x.size(-1)}", (scales.numel() // 16, 1, 1), (x, out, scales))
    return out, scales


def _simt(device):
    arch = target(device)
    return arch == "sm_75" or arch.startswith("gfx10")


def attend(q, k, v, qs, ks, bits, keys, block_q, scale, representatives, causal):
    groups, qp, width = q.shape
    out = torch.empty_like(q, dtype=torch.float16)
    rows = 1 if _simt(q.device) else 32 if torch.version.hip or width == 128 else 64
    name = f"attend{width}_{int(representatives)}{int(causal)}"
    launch(
        q.device,
        name,
        ((qp + rows - 1) // rows, groups, 1),
        (q, k, v, qs, ks, bits, out, qp, k.size(1), keys, block_q, float(scale)),
    )
    return out


def block_indices(keep, qb, kb, causal, *, delta=False):
    b, h, rows, cols = keep.shape
    if keep.dtype not in (torch.bool, torch.uint8, torch.int8):
        keep = keep.bool()
    indices = torch.empty((b, h, rows, cols), dtype=torch.int32, device=keep.device)
    count = torch.empty((b, h, rows), dtype=torch.int32, device=keep.device)
    name = ("lut" if delta else "indices") + str(int(causal))
    launch(
        keep.device,
        name,
        (rows, h, b),
        (keep, indices, count, rows, cols, keep.stride(1), keep.stride(0), keep.stride(2), keep.stride(3), h, qb, kb),
    )
    return indices, count


def attend_blocks(q, k, v, qs, ks, keep, keys, qb, kb, scale, causal):
    groups, qp, width = q.shape
    if kb == 128:
        keep = keep.repeat_interleave(2, -1)[..., : (keys + 63) // 64]
    indices, count = block_indices(keep, qb, 64, causal)
    out = torch.empty_like(q, dtype=torch.float16)
    rows = 1 if _simt(q.device) else 32 if torch.version.hip or width == 128 else 64
    launch(
        q.device,
        f"sparge{width}_{int(causal)}",
        ((qp + rows - 1) // rows, groups, 1),
        (q, k, v, qs, ks, indices, out, qp, k.size(1), keys, qb, float(scale), count),
    )
    return out


def select(qmean, keys, flags, tokens, scale):
    groups, n, d = qmean.shape
    stats = torch.empty((groups, n, n, 3), device=keys.device, dtype=torch.float32)
    bits = torch.empty((groups, n, n, 2), device=keys.device, dtype=torch.int32)
    rows = 1 if _simt(keys.device) else 16
    grid = ((n + rows - 1) // rows, n, groups)

    def run(cut, write):
        launch(
            keys.device,
            f"select{d}_{int(write)}",
            grid,
            (qmean, keys, flags, stats, cut, bits, n, tokens, float(scale)),
        )

    run(stats, False)
    count, total, squared = stats.sum(2).unbind(-1)
    count.clamp_min_(1.0)
    mean = total / count
    cut = mean + 1.125 * (torch.clamp_min(squared / count - mean.square(), 0.0) + 1.0e-6).sqrt()
    run(cut, True)
    return bits
