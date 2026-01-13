
import torch
import numpy as np
import random
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grammar import VOCABULARY, TOKEN_TO_ID
from data.synthetic_data import DataGenerator
from core.model import AlphaSymbolicModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def debug_data_generation():
    print("=== Debugging Data Generation ===")
    gen = DataGenerator(max_depth=4)
    batch = gen.generate_batch(5)
    
    if not batch:
        print("❌ Generated empty batch")
        return False
        
    print(f"Generated {len(batch)} samples")
    for i, item in enumerate(batch):
        print(f"\nSample {i+1}:")
        print(f"  Formula: {item['infix']}")
        print(f"  Tokens: {item['tokens']}")
        print(f"  X stats: min={item['x'].min():.4f}, max={item['x'].max():.4f}, mean={item['x'].mean():.4f}")
        print(f"  Y stats: min={item['y'].min():.4f}, max={item['y'].max():.4f}, mean={item['y'].mean():.4f}")
        
        if np.any(np.isnan(item['y'])) or np.any(np.isinf(item['y'])):
            print("  ❌ NaN/Inf detected in Y")
            return False
            
    return True

def debug_normalization():
    print("\n=== Debugging Normalization ===")
    from ui.app_training import normalize_batch
    
    x = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 10.0, 10.0])] # Case 2 is constant
    y = [np.array([0.0, 5.0, 10.0]), np.array([1.0, 1.0, 1.0])]
    
    norm_x, norm_y = normalize_batch(x, y)
    
    print("Normal Case:")
    print(f"  Orig X: {x[0]}")
    print(f"  Norm X: {norm_x[0]}")
    
    print("Constant Case:")
    print(f"  Orig X: {x[1]}")
    print(f"  Norm X: {norm_x[1]}")
    
    if np.any(np.isnan(norm_x[1])):
        print("  ❌ Constant array caused NaN in normalization")
        return False
    return True

def debug_training_step():
    print("\n=== Debugging Training Step ===")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Init model
    VOCAB_SIZE = len(VOCABULARY)
    model = AlphaSymbolicModel(VOCAB_SIZE + 1, d_model=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # Get batch
    gen = DataGenerator(max_depth=4)
    batch = gen.generate_batch(4)
    from ui.app_training import normalize_batch
    
    x_list = [d['x'] for d in batch]
    y_list = [d['y'] for d in batch]
    x_list, y_list = normalize_batch(x_list, y_list)
    
    token_lists = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in batch]
    max_len = max(len(s) for s in token_lists)
    SOS_ID = VOCAB_SIZE
    
    decoder_input = torch.full((len(batch), max_len + 1), SOS_ID, dtype=torch.long)
    targets = torch.full((len(batch), max_len + 1), -1, dtype=torch.long)
    
    for i, seq in enumerate(token_lists):
        decoder_input[i, 1:len(seq)+1] = torch.tensor(seq, dtype=torch.long)
        targets[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
    decoder_input = decoder_input.to(DEVICE)
    targets = targets.to(DEVICE)
    
    print("Forward pass...")
    logits, _ = model(x_tensor, y_tensor, decoder_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
    
    loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
    print(f"Loss: {loss.item()}")
    
    if torch.isnan(loss):
        print("❌ Loss is NaN")
        return False
        
    loss.backward()
    print("Backward pass successful")
    return True

if __name__ == "__main__":
    set_seed()
    if debug_data_generation() and debug_normalization() and debug_training_step():
        print("\n✅ All debug checks passed!")
    else:
        print("\n❌ Debug checks failed!")
