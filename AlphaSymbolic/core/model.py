import torch
import torch.nn as nn
import numpy as np

class AlphaSymbolicModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, max_seq_len=256, input_dim=2):
        super(AlphaSymbolicModel, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # 1. Point Encoder: Processes pairs/tuples of (x..., y)
        self.point_embedding = nn.Linear(input_dim, d_model)
        
        # We use a standard Transformer Encoder for the "Problem Embedding"
        # Since points are a set, we don't necessarily need positional encoding, 
        # but the Transformer will process them as a sequence.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.problem_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 2. Formula Decoder: Generates tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.formula_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 3. Heads
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Quantiles: 0.25, 0.50, 0.75
        )
        
    def forward(self, x_values, y_values, formula_input, formula_mask=None):
        """
        x_values: [batch, num_points]
        y_values: [batch, num_points]
        formula_input: [batch, seq_len] (Token IDs)
        formula_mask: Optional mask for the decoder (causal mask)
        """
        # -- Problem Encoding --
        # 1. Ensure dimensions
        if x_values.dim() == 2:
            x_values = x_values.unsqueeze(-1) # [batch, num_points, 1]
        
        if y_values.dim() == 2:
            y_values = y_values.unsqueeze(-1) # [batch, num_points, 1]
            
        # 2. Stack x and y: [batch, num_points, n_vars + 1]
        points = torch.cat([x_values, y_values], dim=-1)
        
        # 3. Pad to match input_dim (e.g., 11)
        curr_dim = points.shape[-1]
        if curr_dim < self.input_dim:
            pad_size = self.input_dim - curr_dim
            padding = torch.zeros(points.shape[0], points.shape[1], pad_size, device=points.device)
            points = torch.cat([points, padding], dim=-1)
        
        # Project to d_model
        points_emb = self.point_embedding(points) # [batch, num_points, d_model]
        
        # Encode problem (memory for decoder)
        memory = self.problem_encoder(points_emb)
        
        # -- Formula Decoding --
        # Embed tokens
        tgt = self.token_embedding(formula_input) # [batch, seq_len, d_model]
        tgt = self.pos_encoder(tgt)
        
        # Decode
        # memory is [batch, num_points, d_model]
        # tgt is [batch, seq_len, d_model]
        if formula_mask is None:
             # Create causal mask
            seq_len = formula_input.size(1)
            formula_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(formula_input.device)

        output = self.formula_decoder(tgt, memory, tgt_mask=formula_mask)
        
        # -- Heads --
        # Policy: distribution over vocab for each token position
        logits = self.policy_head(output) # [batch, seq_len, vocab_size]
        
        # Value: estimate value from the LAST token's state
        # (Assuming the last token summarizes the current state)
        last_token_output = output[:, -1, :] # [batch, d_model]
        value = self.value_head(last_token_output) # [batch, 1]
        
        return logits, value

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

if __name__ == "__main__":
    # Smoke Test
    vocab_size = 20
    model = AlphaSymbolicModel(vocab_size=vocab_size, d_model=32)
    
    # Dummy data
    bs = 2
    points = 10
    x = torch.randn(bs, points)
    y = torch.randn(bs, points)
    
    # Formula input (start token + some tokens)
    seq = torch.randint(0, vocab_size, (bs, 5))
    
    logits, value = model(x, y, seq)
    
    print("Logits shape:", logits.shape) # Should be [2, 5, 20]
    print("Value shape:", value.shape)   # Should be [2, 1]
    print("Smoke test passed.")
