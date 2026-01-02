"""
Core state and model management for AlphaSymbolic Gradio App.
"""
import torch
import os
from core.model import AlphaSymbolicModel
from core.grammar import VOCABULARY

# Global state
MODEL = None
DEVICE = None
TRAINING_STATUS = {"running": False, "epoch": 0, "loss": 0, "message": "Listo"}

MODEL_PRESETS = {
    'lite': {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 3, 'num_decoder_layers': 3},
    'pro': {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6}
}
CURRENT_PRESET = 'lite'

def get_device(force_cpu=False):
    """Get the best available device (CUDA > MPS > CPU)."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_device(use_gpu=True):
    """Set the device (GPU or CPU)."""
    global DEVICE, MODEL
    new_device = get_device(force_cpu=not use_gpu)
    
    if MODEL is not None and DEVICE != new_device:
        MODEL = MODEL.to(new_device)
    
    DEVICE = new_device
    return get_device_info()

def get_device_info():
    """Get device info string."""
    global DEVICE
    if DEVICE is None:
        DEVICE = get_device()
    
    if DEVICE.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif DEVICE.type == "mps":
        return "MPS (Apple Silicon)"
    else:
        return "CPU"

def load_model(force_reload=False, preset_name=None):
    """Load or reload the model."""
    global MODEL, DEVICE, CURRENT_PRESET
    
    if preset_name:
        CURRENT_PRESET = preset_name
    
    if DEVICE is None:
        DEVICE = get_device()
    
    VOCAB_SIZE = len(VOCABULARY)
    config = MODEL_PRESETS[CURRENT_PRESET]
    
    print(f"Loading Model [{CURRENT_PRESET.upper()}]...")
    MODEL = AlphaSymbolicModel(
        vocab_size=VOCAB_SIZE + 1, 
        d_model=config['d_model'], 
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'], 
        num_decoder_layers=config['num_decoder_layers']
    ).to(DEVICE)
    
    try:
        state_dict = torch.load("alpha_symbolic_model.pth", map_location=DEVICE, weights_only=True)
        # Check for NaNs
        has_nans = False
        for k, v in state_dict.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                has_nans = True
                break
        
        if has_nans:
            print("⚠️ Modelo corrupto detectado (NaNs). Eliminando archivo y reiniciando pesos.")
            try:
                os.remove("alpha_symbolic_model.pth")
                print("✅ Archivo corrupto eliminado.")
            except OSError as e:
                print(f"Error al eliminar archivo: {e}")
            status = "⚠️ Modelo corrupto eliminado y reiniciado"
        else:
            MODEL.load_state_dict(state_dict)
            MODEL.eval()
            status = f"Modelo cargado ({CURRENT_PRESET})"
    except RuntimeError as e:
        print(f"⚠️ Error de compatibilidad ({e}). Iniciando modelo fresco.")
        status = f"Nuevo modelo ({CURRENT_PRESET})"
    except Exception as e:
        print(f"Error cargando: {e}")
        status = "Sin modelo pre-entrenado"
    
    return status, get_device_info()

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
