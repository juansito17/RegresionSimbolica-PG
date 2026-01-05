"""
Core state and model management for AlphaSymbolic Gradio App.
"""
import torch
import os
from core.model import AlphaSymbolicModel
from core.grammar import VOCABULARY

from collections import deque
import time

# Global state
MODEL = None
DEVICE = None
TRAINING_STATUS = {"running": False, "epoch": 0, "loss": 0, "message": "Listo"}
STOP_TRAINING = False  # Flag to request training stop

def request_stop_training():
    """Request training to stop gracefully."""
    global STOP_TRAINING
    STOP_TRAINING = True
    return "⏹️ Deteniendo entrenamiento..."

def should_stop_training():
    """Check if training should stop."""
    return STOP_TRAINING

def reset_stop_flag():
    """Reset the stop flag (call at start of training)."""
    global STOP_TRAINING
    STOP_TRAINING = False

# Hall of Shame: Rolling buffer of recent failures
# Format: {'time': str, 'target': str, 'predicted': str, 'loss': float, 'stage': str}
TRAINING_ERRORS = deque(maxlen=20)

def add_training_error(target, predicted, loss, stage):
    """Add an error to the Hall of Shame."""
    TRAINING_ERRORS.append({
        'time': time.strftime("%H:%M:%S"),
        'target': target,
        'predicted': predicted,
        'loss': float(loss),
        'stage': stage
    })

def get_training_errors():
    """Get list of errors for the UI."""
    return list(TRAINING_ERRORS)

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
    
    filename = f"alpha_symbolic_model_{CURRENT_PRESET}.pth"
    status = f"Nuevo modelo ({CURRENT_PRESET})" # Default status
    
    # Check Drive for backup IF on colab and main file doesn't exist or is older?
    # Simple strategy: prioritize local, but if local missing, check Drive.
    drive_path = "/content/drive/MyDrive/AlphaSymbolic_Models"
    drive_filename = os.path.join(drive_path, filename)
    
    source_file = None
    if os.path.exists(filename):
        source_file = filename
    elif os.path.exists(drive_filename):
        print(f"📦 Local model missing. Loading from Drive: {drive_filename}")
        source_file = drive_filename

    if source_file:
        try:
            state_dict = torch.load(source_file, map_location=DEVICE, weights_only=True)
            
            # Check for NaNs
            has_nans = False
            for k, v in state_dict.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    has_nans = True
                    break
            
            if has_nans:
                print(f"⚠️ Modelo corrupto detectado (NaNs) en {source_file}. Eliminando.")
                try:
                    os.remove(source_file)
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
    global MODEL, CURRENT_PRESET
    if MODEL is not None:
        filename = f"alpha_symbolic_model_{CURRENT_PRESET}.pth"
        torch.save(MODEL.state_dict(), filename)
        
        # Backup to Google Drive if available
        drive_path = "/content/drive/MyDrive/AlphaSymbolic_Models"
        if os.path.exists("/content/drive"):
            try:
                os.makedirs(drive_path, exist_ok=True)
                drive_filename = os.path.join(drive_path, filename)
                torch.save(MODEL.state_dict(), drive_filename)
                print(f"✅ Modelo respaldado en Drive: {drive_filename}")
            except Exception as e:
                print(f"⚠️ Error al respaldar en Drive: {e}")
