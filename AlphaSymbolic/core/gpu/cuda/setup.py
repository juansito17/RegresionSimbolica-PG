
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Ensure we can find the files
base_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='rpn_cuda_native',
    ext_modules=[
        CUDAExtension(
            name='rpn_cuda_native',
            sources=[
                'bindings.cpp',
                'rpn_kernels.cu',
                'pso_kernels.cu',
                'fused_pso_kernels.cu',
                'decoder.cpp',
                'simplify_kernels.cu',
                'genrand_kernels.cu'
            ],
            extra_compile_args={
                'cxx': ['/O2', '/std:c++17'],
                # -O3: máxima optimización
                # --use_fast_math: instrucciones FP rápidas (rsqrt, fma, etc.)
                # -diag-suppress 221: silencia truncation warning (1e300 -> float32)
                # -Xarch_device/-maxrregcount=64: limita registros para mayor ocupancia
                'nvcc': ['-O3', '--use_fast_math', '-Xcudafe', '--diag_suppress=221',
                         '--maxrregcount=64']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
