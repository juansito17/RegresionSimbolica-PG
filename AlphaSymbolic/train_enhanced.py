"""
Enhanced Training Script for AlphaSymbolic.
Includes:
- Curriculum Learning (simple formulas first)
- Value Network Training (not just policy)
- Proper Loss Weighting
- Regularization (dropout, weight decay)
- Learning Rate Scheduling
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
from model import AlphaSymbolicModel
from synthetic_data import DataGenerator
from grammar import VOCABULARY, TOKEN_TO_ID

def train_enhanced(epochs=1000, batch_size=64, curriculum=True, save_interval=100):
    """
    Enhanced training with curriculum learning and value head training.
    """
    VOCAB_SIZE = len(VOCABULARY)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("AlphaSymbolic Enhanced Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
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
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss_fn = nn.MSELoss()
    
    # SOS token
    SOS_ID = VOCAB_SIZE
    
    # Training loop
    model.train()
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Curriculum: start with simple formulas, increase complexity
        if curriculum:
            progress = min(epoch / (epochs * 0.7), 1.0)  # Ramp up over 70% of training
            current_depth = int(2 + progress * 4)  # Start at depth 2, end at depth 6
        else:
            current_depth = 5
        
        # Generate batch with current difficulty
        data_gen = DataGenerator(max_depth=current_depth)
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
        value_loss = mse_loss_fn(value_pred, value_targets)
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f}) | LR: {lr:.2e} | Depth: {current_depth} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), "alpha_symbolic_model.pth")
                print(f"  -> Saved checkpoint (best loss: {best_loss:.4f})")
    
    # Final save
    torch.save(model.state_dict(), "alpha_symbolic_model.pth")
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Final Loss: {loss.item():.4f}")
    print(f"Model saved to: alpha_symbolic_model.pth")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AlphaSymbolic")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    args = parser.parse_args()
    
    train_enhanced(
        epochs=args.epochs,
        batch_size=args.batch,
        curriculum=not args.no_curriculum
    )
