
import torch
import os

def check_model():
    # Model is in the root directory (parent of utils)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root_dir, "alpha_symbolic_model.pth")
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return

    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        print(f"Loaded {len(state_dict)} keys.")
        
        has_nan = False
        for k, v in state_dict.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"❌ Parameter {k} has NaN/Inf!")
                has_nan = True
        
        if has_nan:
            print("⚠️ Model is CORRUPTED with NaNs.")
            print("Deleting corrupted model file...")
            os.remove(path)
            print("Deleted.")
        else:
            print("✅ Model weights look clean.")
            
    except Exception as e:
        print(f"Error checking model: {e}")

if __name__ == "__main__":
    check_model()
