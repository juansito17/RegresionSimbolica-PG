import os
import sys

_CUDA_DIR = os.path.join(os.path.dirname(__file__), 'cuda')
if _CUDA_DIR not in sys.path:
	sys.path.append(_CUDA_DIR)

from .engine import TensorGeneticEngine
