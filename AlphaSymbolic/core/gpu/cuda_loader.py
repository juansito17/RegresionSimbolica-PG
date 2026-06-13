import glob
import importlib.util
import os
import sys

_RPN_CUDA_NATIVE_CACHE = None
_CUDA_DIR = os.path.join(os.path.dirname(__file__), "cuda")


def load_rpn_cuda_native():
    """Load the repo-local CUDA extension before any stale site-packages build."""
    global _RPN_CUDA_NATIVE_CACHE
    if _RPN_CUDA_NATIVE_CACHE is not None:
        return _RPN_CUDA_NATIVE_CACHE

    cuda_dir = _CUDA_DIR
    if cuda_dir not in sys.path:
        sys.path.insert(0, cuda_dir)

    local_exts = glob.glob(os.path.join(cuda_dir, "rpn_cuda_native*.pyd"))
    if local_exts:
        current = sys.modules.get("rpn_cuda_native")
        current_file = getattr(current, "__file__", "") if current is not None else ""
        if current is not None and os.path.abspath(current_file).startswith(os.path.abspath(cuda_dir)):
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
