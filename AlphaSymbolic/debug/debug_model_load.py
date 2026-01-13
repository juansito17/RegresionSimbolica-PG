
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'AlphaSymbolic'))

from AlphaSymbolic.core.model import AlphaSymbolicModel
from AlphaSymbolic.core.grammar import VOCABULARY

def test_model_load():
    print("--- MODEL LOADING DEBUGGER ---")
    
    # 1. Setup Config similar to app_core
    CURRENT_PRESET = 'lite'
    config = {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 3, 'num_decoder_layers': 3}
    VOCAB_SIZE = len(VOCABULARY)
    
    print(f"Current Vocabulary Size: {VOCAB_SIZE}")
    print(f"Model ID: [LITE]")
    
    # 2. Instantiate Model
    print("Instantiating Model...")
    model = AlphaSymbolicModel(
        vocab_size=VOCAB_SIZE + 1, 
        d_model=config['d_model'], 
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'], 
        num_decoder_layers=config['num_decoder_layers'],
        max_seq_len=256,
        input_dim=11
    )
    
    # 3. Try to load file
    filename = f"alpha_symbolic_model_{CURRENT_PRESET}.pth"
    # Check possible locations
    paths_to_check = [
        os.path.join("AlphaSymbolic", "models", filename),
        os.path.join("models", filename),
        filename
    ]
    
    source_file = None
    for p in paths_to_check:
        if os.path.exists(p):
            source_file = p
            break
            
    if not source_file:
        print(f"❌ Could not find model file: {filename}")
        return

    print(f"Found model file at: {source_file}")
    
    # 4. Load State Dict
    try:
        state_dict = torch.load(source_file, map_location='cpu', weights_only=True)
        print("State dict loaded. Checking keys...")
        
        # Check specific heavy weights for size
        if 'token_embedding.weight' in state_dict:
            print(f"Saved token_embedding shape: {state_dict['token_embedding.weight'].shape}")
            print(f"Current model token_embedding shape: {model.token_embedding.weight.shape}")
            
        print("Attempting load_state_dict...")
        model.load_state_dict(state_dict)
        print("✅ SUCCESS: Model loaded correctly!")
        
    except RuntimeError as e:
        print("\n❌ LOAD FAILED (RuntimeError):")
        print("-" * 40)
        print(str(e))
        print("-" * 40)
        print("\nANALYSIS:")
        if "size mismatch" in str(e):
            print(">> CAUSE: Vocabulary Mismatch.")
            print("The model was trained with a different set of operators/variables.")
            print("Current code has added/removed items from VOCABULARY.")
    except Exception as e:
        print(f"\n❌ LOAD FAILED (General Error): {e}")

if __name__ == "__main__":
    test_model_load()
