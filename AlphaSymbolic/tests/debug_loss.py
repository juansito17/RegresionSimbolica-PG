
import os
import torch
import numpy as np
import random
import sys

# Añadir raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_data import DataGenerator
from ui.app_training import normalize_batch
from core.model import AlphaSymbolicModel
from core.grammar import VOCABULARY, TOKEN_TO_ID

def debug_supervised_batch():
    device = torch.device("cpu")
    vocab_size = len(VOCABULARY)
    SOS_ID = vocab_size
    model = AlphaSymbolicModel(vocab_size=vocab_size+1, input_dim=11).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # Simular lo que hace train_supervised
    batch_size = 8
    point_count = 10
    num_variables = 3 # Multivariable
    
    data_gen = DataGenerator(max_depth=2, num_variables=num_variables)
    batch = data_gen.generate_batch(batch_size, point_count=point_count)
    
    print(f"--- Debugging Batch with {num_variables} variables ---")
    
    x_list = [d['x'] for d in batch]
    y_list = [d['y'] for d in batch]
    
    print(f"Original x shape: {x_list[0].shape}") # (10, 3)
    
    # 1. Normalización
    x_list_norm, y_list_norm = normalize_batch(x_list, y_list)
    print(f"Normalized x shape: {x_list_norm[0].shape}")
    
    # 2. Preparar Tensores
    x_tensor = torch.tensor(np.stack(x_list_norm), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(np.stack(y_list_norm), dtype=torch.float32).to(device)
    if y_tensor.dim() == 2:
        y_tensor = y_tensor.unsqueeze(-1)
    
    print(f"x_tensor shape: {x_tensor.shape}") # (8, 10, 3)
    print(f"y_tensor shape: {y_tensor.shape}") # (8, 10, 1)
    
    # 3. Decoder inputs
    token_lists = [[TOKEN_TO_ID.get(t, TOKEN_TO_ID['C']) for t in d['tokens']] for d in batch]
    max_len = max(len(s) for s in token_lists)
    decoder_input = torch.full((batch_size, max_len + 1), SOS_ID, dtype=torch.long).to(device)
    targets = torch.full((batch_size, max_len + 1), -1, dtype=torch.long).to(device)
    
    for j, seq in enumerate(token_lists):
        decoder_input[j, 1:len(seq)+1] = torch.tensor(seq, dtype=torch.long)
        targets[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    # 4. Forward Pass
    try:
        logits, value = model(x_tensor, y_tensor, decoder_input)
        print(f"Logits shape: {logits.shape}")
        
        # Calculate loss
        loss = ce_loss(logits.view(-1, vocab_size + 1), targets.view(-1))
        print(f"Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Error durante Forward: {e}")

if __name__ == "__main__":
    debug_supervised_batch()
