import sys
import os
import torch
import time
import cProfile
import pstats

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals

def main():
    torch.manual_seed(42)
    GpuGlobals.POPULATION_SIZE = 1000000
    GpuGlobals.GENERATIONS = 50
    GpuGlobals.USE_LEXICASE_SELECTION = False
    
    x = torch.linspace(-10, 10, 200).unsqueeze(1).to("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.sin(x) + x*0.5
    
    engine = TensorGeneticEngine(pop_size=500000)
    
    print("Running warmup...")
    engine.run(x, y, timeout_sec=2)
    
    print("Running profiler for 50 generations...")
    GpuGlobals.GENERATIONS = 50
    
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    
    engine.run(x, y, timeout_sec=20)
    
    profiler.disable()
    end_time = time.time()
    print(f"\nTotal Time (50 gens, 500k pop): {end_time - start_time:.3f}s")
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(50)

if __name__ == "__main__":
    main()
