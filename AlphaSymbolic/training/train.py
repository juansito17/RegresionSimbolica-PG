import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core.model import AlphaSymbolicModel
from data.synthetic_data import DataGenerator
from core.grammar import VOCABULARY, TOKEN_TO_ID, ExpressionTree

def validate(model, val_data, device, vocab_size):
    model.eval()
    total_loss = 0
    total_token_acc = 0
    valid_formulas = 0
    total_samples = len(val_data)
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Prepare batch
    x_list = [d['x'] for d in val_data]
    y_list = [d['y'] for d in val_data]
    token_ids_list = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in val_data]
    
    max_len = max(len(s) for s in token_ids_list)
    SOS_ID = vocab_size
    
    decoder_input = torch.full((total_samples, max_len + 1), SOS_ID, dtype=torch.long)
    targets = torch.full((total_samples, max_len + 1), -1, dtype=torch.long)
    
    for i, seq in enumerate(token_ids_list):
        l = len(seq)
        decoder_input[i, 1:l+1] = torch.tensor(seq, dtype=torch.long)
        targets[i, :l] = torch.tensor(seq, dtype=torch.long)
        
    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(device)
    decoder_input = decoder_input.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        logits, _ = model(x_tensor, y_tensor, decoder_input)
        
        # Loss
        loss = ce_loss_fn(logits.view(-1, vocab_size + 1), targets.view(-1))
        total_loss = loss.item()
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1) # [batch, seq_len]
        
        # Token accuracy (mask padding)
        mask = targets != -1
        correct = (preds == targets) & mask
        total_token_acc = correct.sum().float() / mask.sum().float()
        
        # Formula Validity (reconstruct and check)
        # We need to strip EOS/Padding and stop at first end token if we had one, 
        # but here we just check if the sequence *as predicted* makes sense?
        # Actually, let's just check the ground truth reconstruction for now or 
        # ideally we should run greedy search to generate a formula and check THAT.
        # Checking "teacher forced" predictions for validity is less useful.
        # fast check:
        pass 

    model.train()
    return total_loss, total_token_acc.item()

def train_supervised():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100 
    LR = 1e-4
    VOCAB_SIZE = len(VOCABULARY)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Model
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Data Generator
    data_gen = DataGenerator(max_depth=4)
    
    # Generate Fixed Validation Set
    print("Generating validation set...")
    val_data = data_gen.generate_batch(100) # 100 validation samples
    
    SOS_ID = VOCAB_SIZE
    model.train()
    
    for epoch in range(EPOCHS):
        # 1. Generate Training Batch
        batch_data = data_gen.generate_batch(BATCH_SIZE)
        if not batch_data: continue
        
        # Prepare inputs
        x_list = [d['x'] for d in batch_data]
        y_list = [d['y'] for d in batch_data]
        token_ids_list = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in batch_data]
        
        max_len = max(len(s) for s in token_ids_list)
        
        decoder_input = torch.full((BATCH_SIZE, max_len + 1), SOS_ID, dtype=torch.long)
        targets = torch.full((BATCH_SIZE, max_len + 1), -1, dtype=torch.long)
        
        for i, seq in enumerate(token_ids_list):
            l = len(seq)
            decoder_input[i, 1:l+1] = torch.tensor(seq, dtype=torch.long)
            targets[i, :l] = torch.tensor(seq, dtype=torch.long)
            
        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Forward
        logits, value_pred = model(x_tensor, y_tensor, decoder_input)
        loss = ce_loss_fn(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            val_loss, val_acc = validate(model, val_data, DEVICE, VOCAB_SIZE)
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.2%}")
            
    # Save model
    torch.save(model.state_dict(), "alpha_symbolic_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    try:
        train_supervised()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
