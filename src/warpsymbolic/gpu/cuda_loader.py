import glob
import importlib.util
import os
import sys

_RPN_CUDA_NATIVE_CACHE = None
_CUDA_DIR = os.path.join(os.path.dirname(__file__), "cuda")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_LOCAL_BUILD_DIR = os.path.join(_REPO_ROOT, ".local", "build", "python")


def load_rpn_cuda_native():
    """Load the repo-local CUDA extension before any stale site-packages build."""
    global _RPN_CUDA_NATIVE_CACHE
    if _RPN_CUDA_NATIVE_CACHE is not None:
        return _RPN_CUDA_NATIVE_CACHE

    search_dirs = (_LOCAL_BUILD_DIR, _CUDA_DIR)
    for directory in search_dirs:
        if directory not in sys.path:
            sys.path.insert(0, directory)
        local_exts = []
        for suffix in ("*.pyd", "*.so", "*.dll"):
            local_exts.extend(glob.glob(os.path.join(directory, f"rpn_cuda_native{suffix}")))
        if not local_exts:
            continue

        current = sys.modules.get("rpn_cuda_native")
        current_file = getattr(current, "__file__", "") if current is not None else ""
        if current is not None and os.path.abspath(current_file).startswith(os.path.abspath(directory)):
            return current

        spec = importlib.util.spec_from_file_location("rpn_cuda_native", local_exts[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules["rpn_cuda_native"] = module
        spec.loader.exec_module(module)
        _RPN_CUDA_NATIVE_CACHE = module
        return module

    import rpn_cuda_native
    _RPN_CUDA_NATIVE_CACHE = rpn_cuda_native
    return rpn_cuda_native
