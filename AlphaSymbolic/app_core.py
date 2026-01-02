"""
Core state and model management for AlphaSymbolic Gradio App.
"""
import torch
from model import AlphaSymbolicModel
from grammar import VOCABULARY

# Global state
MODEL = None
DEVICE = None
TRAINING_STATUS = {"running": False, "epoch": 0, "loss": 0, "message": "Listo"}

def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(force_reload=False):
    """Load or reload the model."""
    global MODEL, DEVICE
    
    DEVICE = get_device()
    VOCAB_SIZE = len(VOCABULARY)
    
    MODEL = AlphaSymbolicModel(
        vocab_size=VOCAB_SIZE + 1, 
        d_model=128, 
        nhead=4,
        num_encoder_layers=3, 
        num_decoder_layers=3
    ).to(DEVICE)
    
    try:
        MODEL.load_state_dict(torch.load("alpha_symbolic_model.pth", map_location=DEVICE, weights_only=True))
        MODEL.eval()
        status = "✅ Modelo cargado"
    except:
        status = "⚠️ Sin modelo pre-entrenado"
    
    device_info = f"🖥️ {DEVICE.type.upper()}"
    if DEVICE.type == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"
    
    return status, device_info

def get_model():
    """Get the current model, loading if needed."""
    global MODEL, DEVICE
    if MODEL is None:
        load_model()
    return MODEL, DEVICE

def save_model():
    """Save the current model."""
    global MODEL
    if MODEL is not None:
        torch.save(MODEL.state_dict(), "alpha_symbolic_model.pth")
