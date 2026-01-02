import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import AlphaSymbolicModel
from synthetic_data import DataGenerator
from grammar import VOCABULARY, TOKEN_TO_ID

def train_supervised():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100 # Or iterations
    LR = 1e-4
    VOCAB_SIZE = len(VOCABULARY)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Model
    model = AlphaSymbolicModel(vocab_size=VOCABULARY_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss functions
    # Policy: Cross Entropy (predict next token)
    # Value: MSE (predict negative RMSE) -- though in supervised we focus on Policy primarily
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # Assuming padding is -1, but we'll handle masks
    
    # Data Generator
    data_gen = DataGenerator(max_depth=4)
    
    # Special Tokens
    # We might need a SOS (Start of Sequence) token and EOS.
    # For now, let's assume the first token generated is part of the formula.
    # To properly train, we usually prepend a <SOS> token.
    # Let's add <SOS> to grammar dynamically or just hack it here.
    # Hack: We will use a dedicated ID for SOS that is out of range of normal vocab, or simplest: use index 0 as SOS if we shift everything.
    # Better: Update vocab in grammar.py? Or just assume the model predicts the first token from "empty" input?
    # Standard Seq2Seq: Input to decoder is <SOS> + tokens[:-1]. Target is tokens.
    
    # Let's add a dummy SOS concept: Input is padded with a start token.
    SOS_ID = VOCAB_SIZE # Temporary ID for SOS
    # Note: Model embedding layer needs to accommodate SOS_ID.
    # Let's re-init model with VOCAB_SIZE + 1
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=64).to(DEVICE)
    
    model.train()
    
    for epoch in range(EPOCHS):
        # 1. Generate Batch
        batch_data = data_gen.generate_batch(BATCH_SIZE)
        if not batch_data: continue
        
        # Prepare inputs
        x_list = [d['x'] for d in batch_data]
        y_list = [d['y'] for d in batch_data]
        
        # Convert tokens to IDs
        token_ids_list = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in batch_data]
        
        # Pad sequences
        max_len = max(len(s) for s in token_ids_list)
        # Input: SOS + sequence
        # Target: sequence + EOS (optional) or just sequence
        
        # Let's do: Decoder Input = [SOS, t1, t2...]
        # Target = [t1, t2, ..., EOS] (or stop at last token)
        
        decoder_input = torch.full((BATCH_SIZE, max_len + 1), SOS_ID, dtype=torch.long)
        targets = torch.full((BATCH_SIZE, max_len + 1), -1, dtype=torch.long) # -1 padding
        
        for i, seq in enumerate(token_ids_list):
            l = len(seq)
            decoder_input[i, 1:l+1] = torch.tensor(seq, dtype=torch.long)
            targets[i, :l] = torch.tensor(seq, dtype=torch.long)
            # Add EOS if we had one. For now train to reproduce sequence.
            # Actually, standard is: Input [SOS, A, B], Target [A, B, EOS]
            
        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Forward
        # Create mask for decoder (causal)
        # But for training we can just use the internal mask generation or provide one.
        # The model's forward generates a square mask if None is provided.
        
        logits, value_pred = model(x_tensor, y_tensor, decoder_input)
        
        # Logits: [batch, seq_len, vocab_size]
        # Targets: [batch, seq_len]
        
        # Flatten for loss
        # We want to predict targets[i] from decoder_input[i]
        # decoder_input[:, 0] is SOS -> predicts targets[:, 0] (which is first token)
        
        # Truncate logits to match targets length if needed, or vice versa
        # logits predicts next token for every input token.
        # Input: [SOS, A, B]
        # Logits at index 0 (SOS) -> Predict A (targets[0])
        # Logits at index 1 (A)   -> Predict B (targets[1])
        
        # So we align:
        # Preds = logits[:, :-1, :] (Drop prediction for last input if it's padding/EOS/extra)
        # But wait, decoder_input has length max_len+1. Targets has length max_len+1.
        
        # Simply:
        # Logits shape is [Batch, L, V]
        # Target shape is [Batch, L]
        
        loss = ce_loss_fn(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    # Save model
    torch.save(model.state_dict(), "alpha_symbolic_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    try:
        VOCABULARY_SIZE = len(VOCABULARY)
        train_supervised()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
