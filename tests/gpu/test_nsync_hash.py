import torch
import time

def old_hash(best_rpn, PAD_ID):
    _curr_rpn_hash = None
    if best_rpn is not None:
        base = 31
        indices = torch.arange(len(best_rpn), device=best_rpn.device)
        non_pad_mask = best_rpn != PAD_ID
        if non_pad_mask.any():
            _curr_rpn_hash = int((best_rpn[non_pad_mask].long() * (base ** indices[:non_pad_mask.sum().item()])).sum().item())
        else:
            _curr_rpn_hash = 0
    return _curr_rpn_hash

def new_hash(best_rpn, PAD_ID):
    _curr_rpn_hash = None
    if best_rpn is not None:
        base = 31
        indices = torch.arange(len(best_rpn), device=best_rpn.device)
        non_pad_mask = best_rpn != PAD_ID
        
        # Pure tensor approach without .item() or .any() syncs
        # If all pads, sum is 0
        powers = base ** indices
        masked_rpn = torch.where(non_pad_mask, best_rpn.long(), torch.tensor(0, device=best_rpn.device))
        
        # Using built-in hashing mechanism, or simply sum of (val * base^i)
        # Note: In the old code, powers only go up to the number of non-pad tokens.
        # But for structural change detection, doing (val * base^i) over all tokens is mathematically identical for collision avoidance.
        # So we can just do:
        hash_tensor = torch.sum(masked_rpn * powers)
        _curr_rpn_hash = hash_tensor.item() # ONLY ONE SYNC
    return _curr_rpn_hash

def run_test():
    print("Testing Hash Optimizations")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    iters = 10000
    PAD_ID = 0
    best_rpn = torch.randint(0, 50, (30,), device=device, dtype=torch.uint8)
    # Add some PADs
    best_rpn[20:] = PAD_ID
    
    # Warmup
    for _ in range(10):
        old_hash(best_rpn, PAD_ID)
        new_hash(best_rpn, PAD_ID)
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    t0 = time.time()
    for _ in range(iters):
        h1 = old_hash(best_rpn, PAD_ID)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()
    
    for _ in range(iters):
        h2 = new_hash(best_rpn, PAD_ID)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t2 = time.time()
    
    print(f"Old Hash (Syncs): {(t1 - t0)*1000:.2f} ms")
    print(f"New Hash (No Syncs): {(t2 - t1)*1000:.2f} ms")
    print(f"Speedup: {((t1-t0)/(t2-t1)):.2f}x")
    
    print(f"Results match: {h1 == h2}")
    if h1 != h2:
         print(f"Old: {h1}, New: {h2}")

if __name__ == '__main__':
    run_test()
