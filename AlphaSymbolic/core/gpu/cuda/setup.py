
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
                'decoder.cpp',
                'simplify_kernels.cu',
                'genrand_kernels.cu'
            ],
            extra_compile_args={
                'cxx': ['/O2', '/std:c++17'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
