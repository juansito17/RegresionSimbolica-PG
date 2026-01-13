
import os
import torch
import sys

# Añadir el raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.app_core import load_model, MODEL_PRESETS, get_model

def test_model_initialization():
    print("==========================================")
    print("   TEST: INICIALIZACIÓN DE MODELO NUEVO   ")
    print("==========================================")
    
    preset = 'lite'
    filename = f"alpha_symbolic_model_{preset}.pth"
    local_path = os.path.join("models", filename)
    backup_path = local_path + ".bak"
    
    # 1. Simular que no hay modelo guardado
    already_exists = False
    if os.path.exists(local_path):
        print(f"📦 Moviendo modelo existente a {backup_path}...")
        os.rename(local_path, backup_path)
        already_exists = True
    else:
        print("✅ No se detectó modelo previo.")

    try:
        # 2. Intentar cargar el modelo (debería crear uno nuevo)
        print(f"🔄 Llamando a load_model(preset_name='{preset}')...")
        status, device_info = load_model(preset_name=preset)
        
        print(f"📊 Estado devuelto: {status}")
        print(f"💻 Dispositivo: {device_info}")
        
        # 3. Verificar resultados
        model, device = get_model()
        
        if model is not None:
            print("✅ Verificación: El objeto MODEL no es None.")
            # Verificar si los pesos son aleatorios (una forma simple es ver si están cerca de 0)
            first_param = next(model.parameters())
            print(f"📝 Ejemplo de peso inicial: {first_param[0][0].item():.6f}")
            
            if "Nuevo modelo" in status:
                print("🎯 TEST PASADO: El sistema identificó correctamente un modelo nuevo.")
            else:
                print("❌ ERROR: El estado no indica que sea un modelo nuevo.")
        else:
            print("❌ ERROR: El modelo es None.")
            
    finally:
        # 4. Restaurar el modelo original si existía
        if already_exists:
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(backup_path, local_path)
            print("📦 Modelo original restaurado.")

if __name__ == "__main__":
    test_model_initialization()
