"""CPU / GPU array backend.

Field math is written against numpy; on the GPU the same code runs on CuPy arrays (CuPy
implements numpy's ufunc and array-function protocols, so `np.cos(cupy_array)` runs on the GPU).
Distance fields on the GPU use NVIDIA Warp (engine/sdf_gpu.py). Caches always store host (numpy)
arrays, so results are interchangeable between the two backends.

The GPU is used when available unless switched off with set_gpu(False) or the environment
variable DESIGN_ENGINE_GPU=0.
"""
import os
import warnings

import numpy as np

_cp = None
_available = None
_enabled = os.environ.get("DESIGN_ENGINE_GPU", "1") not in ("0", "false", "off")


def _load():
    global _cp, _available
    if _available is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "CUDA path could not be detected": libs come from pip
                import cupy
            _available = cupy.cuda.runtime.getDeviceCount() > 0
            _cp = cupy if _available else None
        except Exception:
            _available = False
    return _available


def gpu_available():
    return _load()


def gpu_enabled():
    return _enabled and gpu_available()


def set_gpu(on):
    global _enabled
    _enabled = bool(on)


def cupy():
    _load()
    return _cp


def xp_of(a):
    """numpy or cupy, whichever owns array `a`."""
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp
    return np


def to_host(a):
    """numpy array (copies from the GPU if needed)."""
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return a


def to_device(a, xp):
    """Array on xp's device (no copy if it is already there)."""
    if xp is np:
        return to_host(a)
    return xp.asarray(a)


_nvml = None


def gpu_memory():
    """(free, total) memory of the whole graphics card in bytes, or (0, 0) without a GPU.

    Uses NVML (same numbers as nvidia-smi). CUDA's own memGetInfo cannot be trusted on Windows:
    under the WDDM display driver it reports a per-process view (it showed 1.2 GB in use while
    the card actually had 10.7 GB in use)."""
    global _nvml
    if not gpu_available():
        return 0, 0
    try:
        if _nvml is None:
            import pynvml
            pynvml.nvmlInit()
            _nvml = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        info = _nvml[0].nvmlDeviceGetMemoryInfo(_nvml[1])
        return int(info.free), int(info.total)
    except Exception:
        free, total = _cp.cuda.runtime.memGetInfo()
        return int(free), int(total)


GPU_WORKER_BYTES = int(0.8 * 2**30)   # measured: CUDA context + one petal's arrays, per process
GPU_HEADROOM = int(1.5 * 2**30)       # left free for other programs (display, browser, CAD)
# Below this the GPU is no faster (Draft lamp edits: GPU 1.6 s, CPU 1.5 s) but a CUDA context
# costs each process ~0.35 GB RAM and ~0.5 GB GPU memory, so small grids stay on the CPU.
GPU_MIN_VOXELS = 1_000_000


def use_gpu_for(voxels):
    # size first: asking about the GPU loads CuPy and starts CUDA, which small jobs should not
    return voxels >= GPU_MIN_VOXELS and gpu_enabled()


def gpu_worker_count(requested, running=0, extra=0):
    """How many worker processes may use the GPU at once, given the card's free memory now.
    `running` = workers already holding a GPU context (their memory is already counted as used);
    `extra` = GPU bytes each worker needs on top of the usual (e.g. a big part's search tree)."""
    free, _ = gpu_memory()
    return max(1, min(requested, running + int((free - GPU_HEADROOM) // (GPU_WORKER_BYTES + extra))))


def is_out_of_memory(err):
    """True for CUDA out-of-memory errors from CuPy or Warp."""
    text = f"{type(err).__name__}: {err}".lower()
    return "outofmemory" in text.replace(" ", "") or "out of memory" in text


def release_gpu_memory():
    """Give memory CuPy keeps pooled for reuse back to the driver (so other programs can use it)."""
    if _cp is not None:
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()


def gaussian_filter(xp):
    if xp is np:
        from scipy.ndimage import gaussian_filter as f
    else:
        from cupyx.scipy.ndimage import gaussian_filter as f
    return f
