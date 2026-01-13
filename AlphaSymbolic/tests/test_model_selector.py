
import sys
import os
import torch
sys.path.append(os.getcwd())

from ui.app_core import load_model, get_model, MODEL_PRESETS, CURRENT_PRESET

def test_model_switching():
    print("Testing Model Selector...")
    
    # Test 1: Switch to Lite
    print("\n[Step 1] Switching to 'lite'...")
    status, _ = load_model(preset_name='lite')
    model, _ = get_model()
    
    # Verify Lite Config
    expected_lite = MODEL_PRESETS['lite']
    
    # Introspect model
    actual_d_model = model.d_model
    actual_layers = len(model.problem_encoder.layers)
    actual_nhead = model.problem_encoder.layers[0].self_attn.num_heads
    
    is_lite_correct = (
        actual_d_model == expected_lite['d_model'] and
        actual_nhead == expected_lite['nhead'] and
        actual_layers == expected_lite['num_encoder_layers']
    )
    
    print(f"  Status: {status}")
    print(f"  Config matches Lite? {is_lite_correct}")
    print(f"  (Expected: {expected_lite})")
    print(f"  (Got: d_model={actual_d_model}, nhead={actual_nhead}, layers={actual_layers})")
    
    if not is_lite_correct:
        print("FAILED: Lite model did not load correctly.")
        return False

    # Test 2: Switch to Pro
    print("\n[Step 2] Switching to 'pro'...")
    status, _ = load_model(preset_name='pro')
    model, _ = get_model()
    
    # Introspect model
    actual_d_model = model.d_model
    actual_layers = len(model.problem_encoder.layers)
    actual_nhead = model.problem_encoder.layers[0].self_attn.num_heads
    
    expected_pro = MODEL_PRESETS['pro']
    is_pro_correct = (
        actual_d_model == expected_pro['d_model'] and
        actual_nhead == expected_pro['nhead'] and
        actual_layers == expected_pro['num_encoder_layers']
    )
    
    print(f"  Status: {status}")
    print(f"  Config matches Pro? {is_pro_correct}")
    print(f"  (Expected: {expected_pro})")
    print(f"  (Got: d_model={actual_d_model}, nhead={actual_nhead}, layers={actual_layers})")

    if not is_pro_correct:
        print("FAILED: Pro model did not load correctly.")
        return False
        
    return True

if __name__ == "__main__":
    if test_model_switching():
        print("\n✅ MODEL SELECTOR VERIFIED.")
    else:
        print("\n❌ MODEL SELECTOR FAILED.")
