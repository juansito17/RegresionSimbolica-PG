
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Ensure we can find the files
base_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='rpn_cuda',
    ext_modules=[
        CUDAExtension('rpn_cuda', [
            os.path.join(base_path, 'rpn_kernels.cu'),
            os.path.join(base_path, 'bindings.cpp'),
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
