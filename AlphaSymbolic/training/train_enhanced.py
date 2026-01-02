"""
Enhanced Training Script for AlphaSymbolic.
Includes:
- Curriculum Learning (simple formulas first, then complex operators)
- Value Network Training (not just policy)
- Proper Loss Weighting
- Regularization (dropout, weight decay)
- Learning Rate Scheduling (OneCycleLR)
- Gradient Accumulation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import time

from core.model import AlphaSymbolicModel
from data.synthetic_data import DataGenerator
from core.grammar import VOCABULARY, TOKEN_TO_ID

def train_enhanced(epochs=1000, batch_size=64, curriculum=True, save_interval=100, accum_steps=4):
    """
    Enhanced training with curriculum learning, value head, OneCycleLR, and Gradient Accumulation.
    effective_batch_size = batch_size * accum_steps
    """
    VOCAB_SIZE = len(VOCABULARY)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("AlphaSymbolic Enhanced Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Epochs: {epochs}")
    print(f"Physical Batch Size: {batch_size}")
    print(f"Accumulation Steps: {accum_steps}")
    print(f"Effective Batch Size: {batch_size * accum_steps}")
    print(f"Curriculum Learning: {curriculum}")
    
    # Model with dropout
    model = AlphaSymbolicModel(
        vocab_size=VOCAB_SIZE + 1, 
        d_model=128,  # Larger model
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) # Higher LR for OneCycle
    
    # Learning rate scheduler
    # Steps per epoch is actually 1 because we generate data on fly? 
    # No, usually OneCycle needs total steps: epochs * steps_per_epoch
    # Here one "epoch" is one batch generation loop?
    # In the original code, `range(epochs)` ran one batch per loop iteration.
    # So total_steps = epochs.
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=epochs, 
        pct_start=0.3, 
        div_factor=25, 
        final_div_factor=1000
    )
    
    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss_fn = nn.MSELoss()
    
    # SOS token
    SOS_ID = VOCAB_SIZE
    
    # Curriculum Levels
    # Level 0: Basic arithmetic
    # Level 1: Division included
    # Level 2: All (Trig, Exp, Log)
    op_levels = [
        ['+', '-', '*'],
        ['+', '-', '*', '/'],
        None # All
    ]
    
    # Training loop
    model.train()
    start_time = time.time()
    best_loss = float('inf')
    
    optimizer.zero_grad() # Initialize gradients to zero before the loop
    
    for epoch in range(epochs):
        # Determine Curriculum Level
        if curriculum:
            progress = epoch / epochs
            # Depth: 2 -> 6
            current_depth = int(2 + progress * 4)
            # Operators
            if progress < 0.3:
                current_ops = op_levels[0]
            elif progress < 0.6:
                current_ops = op_levels[1]
            else:
                current_ops = op_levels[2]
        else:
            current_depth = 5
            current_ops = None
        
        # Generate batch with current difficulty
        data_gen = DataGenerator(max_depth=current_depth, allowed_operators=current_ops)
        batch_data = data_gen.generate_batch(batch_size)
        
        if len(batch_data) < batch_size // 2:
            continue
        
        actual_batch = len(batch_data)
        
        # Prepare data
        x_list = [d['x'] for d in batch_data]
        y_list = [d['y'] for d in batch_data]
        token_ids_list = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in batch_data]
        
        # Pad sequences
        max_len = max(len(s) for s in token_ids_list)
        
        # Decoder input: [SOS, tokens...]
        decoder_input = torch.full((actual_batch, max_len + 1), SOS_ID, dtype=torch.long)
        targets = torch.full((actual_batch, max_len + 1), -1, dtype=torch.long)  # -1 = padding
        
        # Target values (negative normalized RMSE, scaled to [-1, 1])
        # For synthetic data, the "perfect" formula exists, so we use length penalty as proxy
        value_targets = torch.zeros(actual_batch, 1)
        
        for i, seq in enumerate(token_ids_list):
            l = len(seq)
            decoder_input[i, 1:l+1] = torch.tensor(seq, dtype=torch.long)
            targets[i, :l] = torch.tensor(seq, dtype=torch.long)
            # Value target: shorter formulas are "better" (simpler)
            value_targets[i] = -len(seq) / 20.0  # Normalize roughly to [-1, 0]
        
        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        targets = targets.to(DEVICE)
        value_targets = value_targets.to(DEVICE)
        
        # Forward pass
        logits, value_pred = model(x_tensor, y_tensor, decoder_input)
        
        # Policy loss (cross-entropy)
        policy_loss = ce_loss_fn(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
        
        # Value loss (MSE)
        # Value loss (ensure dimensions match)
        if value_pred.shape != value_targets.shape:
             value_pred = value_pred.view_as(value_targets)
             
        value_loss = mse_loss_fn(value_pred, value_targets)
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss
        
        # Gradient Accumulation
        # Normalize loss by accum_steps to keep magnitude same
        loss = loss / accum_steps
        loss.backward()
        
        if (epoch + 1) % accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging (multiply loss back by accum_steps for display)
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            real_loss = loss.item() * accum_steps
            ops_name = "All" if current_ops is None else str(len(current_ops))
            print(f"Epoch {epoch:4d} | Loss: {real_loss:.4f} (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f}) | LR: {lr:.2e} | Depth: {current_depth} | Ops: {ops_name} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            real_loss = loss.item() * accum_steps
            if real_loss < best_loss:
                best_loss = real_loss
                torch.save(model.state_dict(), "alpha_symbolic_model.pth")
                print(f"  -> Saved checkpoint (best loss: {best_loss:.4f})")
    
    # Final save
    torch.save(model.state_dict(), "alpha_symbolic_model.pth")
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Model saved to: alpha_symbolic_model.pth")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AlphaSymbolic")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (physical)")
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    args = parser.parse_args()
    
    train_enhanced(
        epochs=args.epochs,
        batch_size=args.batch,
        curriculum=not args.no_curriculum,
        accum_steps=args.accum
    )

