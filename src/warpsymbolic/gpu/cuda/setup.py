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
                'genrand_kernels.cu',
                'backward_kernels.cu',
                'diversity_kernels.cu',
                'lbfgs_kernels.cu',       # L-BFGS-B optimizer kernel
                'best_tracker_kernels.cu'  # Best tracking kernel
            ],
            depends=['eval_core.cuh'],
            extra_compile_args={
                'cxx': ['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],
                # -O3: máxima optimización.
                # No --use_fast_math: it maps sinf/cosf/expf/logf/powf to the
                # hardware intrinsics, whose error grows with |x| (sin(1e4) is
                # off by ~1e-3 relative), so GPU fitness disagreed with the
                # final strict/NumPy validation. We keep the cheap parts of
                # fast-math (flush-to-zero, approximate div/sqrt, FMA) and the
                # accurate transcendental functions.
                # -diag-suppress 221: silencia truncation warning (1e300 -> float32)
                # Do not impose one register cap on every kernel. The RPN evaluator,
                # PSO and L-BFGS have very different register/occupancy trade-offs;
                # a global maxrregcount caused local-memory spills on the RTX 3050.
                'nvcc': ['-O3', '-ftz=true', '-prec-div=false', '-prec-sqrt=false', '-fmad=true',
                         '-Xcudafe', '--diag_suppress=221']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
