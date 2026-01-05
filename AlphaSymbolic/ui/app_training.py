"""
Training functions for AlphaSymbolic Gradio App.
With proper data normalization.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from collections import deque
import random
import time

from core.grammar import VOCABULARY, TOKEN_TO_ID, OPERATORS, OPERATOR_STAGES
from data.synthetic_data import DataGenerator
from ui.app_core import get_model, save_model, TRAINING_STATUS, add_training_error, should_stop_training, reset_stop_flag
from core.loss import QuantileLoss


def get_allowed_token_mask(stage, vocab_size, device):
    """
    Creates a mask tensor for token logits.
    Allowed tokens = 1.0, Disallowed = 0.0 (for multiplication mask)
    Or returns indices of allowed tokens for -inf masking.
    """
    allowed_ops = OPERATOR_STAGES.get(stage, list(OPERATORS.keys()))
    
    # All terminals are always allowed
    allowed_tokens = set(['x', 'C', '0', '1', '2', '3', '5', '10', 'pi', 'e'])
    allowed_tokens.update(allowed_ops)
    
    # Build mask
    mask = torch.zeros(vocab_size + 1, device=device)  # +1 for SOS token
    for token in allowed_tokens:
        if token in TOKEN_TO_ID:
            mask[TOKEN_TO_ID[token]] = 1.0
    mask[vocab_size] = 1.0  # SOS always allowed
    
    return mask


def normalize_batch(x_list, y_list):
    """Normalize X and Y values to prevent numerical instability."""
    normalized_x = []
    normalized_y = []
    
    for x, y in zip(x_list, y_list):
        # Normalize X to [-1, 1]
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-6:
            x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            x_norm = np.zeros_like(x)
        
        # Normalize Y to [-1, 1] 
        y_min, y_max = y.min(), y.max()
        if y_max - y_min > 1e-6:
            y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        else:
            y_norm = np.zeros_like(y)
        
        normalized_x.append(x_norm)
        normalized_y.append(y_norm)
    
    return normalized_x, normalized_y


def train_basic(epochs, batch_size, point_count=10, progress=gr.Progress()):
    """Basic training with synthetic data."""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    
    try:
        MODEL, DEVICE = get_model()
        
        MODEL.train()
        optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-6)
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
        VOCAB_SIZE = len(VOCABULARY)
        SOS_ID = VOCAB_SIZE
        
        data_gen = DataGenerator(max_depth=4)
        losses = []
        
        for epoch in range(int(epochs)):
            progress((epoch + 1) / epochs, desc=f"Epoca {epoch+1}/{int(epochs)} [{DEVICE.type.upper()}]")
            
            # Mix of inverse (known formulas) + random data (AlphaTensor-style)
            half_batch = int(batch_size) // 2
            batch_inverse = data_gen.generate_inverse_batch(half_batch, point_count=int(point_count))
            batch_random = data_gen.generate_batch(int(batch_size) - half_batch, point_count=int(point_count))
            batch = batch_inverse + batch_random
            if len(batch) < 2:
                continue
            
            x_list = [d['x'] for d in batch]
            y_list = [d['y'] for d in batch]
            
            # Normalize data
            x_list, y_list = normalize_batch(x_list, y_list)
            
            token_lists = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in batch]
            
            max_len = max(len(s) for s in token_lists)
            decoder_input = torch.full((len(batch), max_len + 1), SOS_ID, dtype=torch.long)
            targets = torch.full((len(batch), max_len + 1), -1, dtype=torch.long)
            
            for i, seq in enumerate(token_lists):
                decoder_input[i, 1:len(seq)+1] = torch.tensor(seq, dtype=torch.long)
                targets[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            
            x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
            y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            logits, _ = MODEL(x_tensor, y_tensor, decoder_input)
            loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
            
            # Skip if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
        
        save_model()
        MODEL.eval()
        TRAINING_STATUS["running"] = False
        
        if not losses:
            return "Error: No se pudo calcular loss (revisar datos)", None
        
        fig = create_loss_plot(losses, "Entrenamiento Basico")
        
        result = f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #4ade80;">
            <h2 style="color: #4ade80; margin: 0;">Entrenamiento Completado</h2>
            <p style="color: white;">Epocas: {int(epochs)} | Loss Final: {losses[-1]:.4f}</p>
            <p style="color: #00d4ff;">Dispositivo: {DEVICE.type.upper()}</p>
        </div>
        """
        return result, fig
        
    except Exception as e:
        TRAINING_STATUS["running"] = False
        return f"Error: {str(e)}", None


def train_curriculum(epochs, batch_size, point_count=10, progress=gr.Progress()):
    """Curriculum Learning - starts simple, increases difficulty gradually."""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    
    try:
        MODEL, DEVICE = get_model()
        
        MODEL.train()
        optimizer = torch.optim.AdamW(MODEL.parameters(), lr=5e-5, weight_decay=0.01)  # Lower LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
        VOCAB_SIZE = len(VOCABULARY)
        SOS_ID = VOCAB_SIZE
        losses = []
        
        for epoch in range(int(epochs)):
            # Curriculum: slow progression
            # Stage 1 (0-50%): depth 2-3, 80% inverse data
            # Stage 2 (50-80%): depth 3-4, 50% inverse data  
            # Stage 3 (80-100%): depth 4-5, 20% inverse data
            progress_pct = epoch / epochs
            
            if progress_pct < 0.5:
                current_depth = 2 + int(progress_pct * 2)  # 2-3
                inverse_ratio = 0.8
            elif progress_pct < 0.8:
                current_depth = 3 + int((progress_pct - 0.5) * 3.3)  # 3-4
                inverse_ratio = 0.5
            else:
                current_depth = 4 + int((progress_pct - 0.8) * 5)  # 4-5
                inverse_ratio = 0.2
            
            progress((epoch + 1) / epochs, desc=f"Epoca {epoch+1}/{int(epochs)} (prof: {current_depth}, inv: {inverse_ratio:.0%}) [{DEVICE.type.upper()}]")
            
            data_gen = DataGenerator(max_depth=current_depth)
            
            # Mix inverse + random based on curriculum stage
            n_inverse = int(batch_size * inverse_ratio)
            n_random = int(batch_size) - n_inverse
            
            batch_inverse = data_gen.generate_inverse_batch(max(1, n_inverse), point_count=int(point_count)) if n_inverse > 0 else []
            batch_random = data_gen.generate_batch(max(1, n_random), point_count=int(point_count)) if n_random > 0 else []
            batch = batch_inverse + batch_random
            if len(batch) < 2:
                continue
            
            x_list = [d['x'] for d in batch]
            y_list = [d['y'] for d in batch]
            x_list, y_list = normalize_batch(x_list, y_list)
            
            token_lists = [[TOKEN_TO_ID[t] for t in d['tokens']] for d in batch]
            
            max_len = max(len(s) for s in token_lists)
            decoder_input = torch.full((len(batch), max_len + 1), SOS_ID, dtype=torch.long)
            targets = torch.full((len(batch), max_len + 1), -1, dtype=torch.long)
            
            for i, seq in enumerate(token_lists):
                decoder_input[i, 1:len(seq)+1] = torch.tensor(seq, dtype=torch.long)
                targets[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            
            x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
            y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits, value_pred = MODEL(x_tensor, y_tensor, decoder_input)
            
            # Policy Loss
            loss_policy = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
            
            # Value Loss
            # For supervised learning, these are "perfect" solutions, so Value Target = 1.0
            value_targets = torch.ones_like(value_pred)
            loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
            
            # Combined Loss
            loss = loss_policy + 0.5 * loss_value
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
        
        save_model()
        MODEL.eval()
        TRAINING_STATUS["running"] = False
        
        if not losses:
            return "Error: No se pudo calcular loss", None
        
        fig = create_loss_plot(losses, "Curriculum Learning")
        
        result = f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #00d4ff;">
            <h2 style="color: #00d4ff; margin: 0;">Curriculum Learning Completado</h2>
            <p style="color: white;">Epocas: {int(epochs)} | Loss Final: {losses[-1]:.4f}</p>
            <p style="color: #888;">Profundidad maxima: 6 | Dispositivo: {DEVICE.type.upper()}</p>
        </div>
        """
        return result, fig
        
    except Exception as e:
        TRAINING_STATUS["running"] = False
        return f"Error: {str(e)}", None


def train_self_play(iterations, problems_per_iter, point_count=10, progress=gr.Progress()):
    """AlphaZero Self-Play loop."""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    reset_stop_flag()  # Reset stop flag at start
    
    try:
        MODEL, DEVICE = get_model()
        
        from search.mcts import MCTS
        
        optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
        # Scheduler: Reduce LR when plateauing to help convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
        
        # Losses for AlphaZero
        # Policy: KLDiv (comparing distributions)
        # Value: Quantile Loss (3 Quantiles)
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        quantile_loss_fn = QuantileLoss()
        
        VOCAB_SIZE = len(VOCABULARY)
        SOS_ID = VOCAB_SIZE
        
        replay_buffer = deque(maxlen=20000)
        
        # Adaptive Curriculum State
        current_depth = 2
        data_gen = DataGenerator(max_depth=current_depth)
        
        # MCTS for A100: Increase batch size and simulations significantly
        # Adjusted for RTX 3050/i5: Batch 64 is smoother (less CPU wait)
        searcher = MCTS(MODEL, DEVICE, max_simulations=500, complexity_lambda=0.1, batch_size=64)
        
        rmses = []
        losses = []
        best_avg_rmse = float('inf')
        
        start_time = time.time()
        
        for iteration in range(int(iterations)):
            # Check for stop request
            if should_stop_training():
                print("⏹️ Training stopped by user")
                break
            # ETA Calculation
            elapsed = time.time() - start_time
            if iteration > 0:
                avg_time_per_iter = elapsed / iteration
                remaining_iters = int(iterations) - iteration
                eta_seconds = remaining_iters * avg_time_per_iter
                
                # Format ETA
                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds/60:.0f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "Calculando..."

            # Adaptive Curriculum Check
            # Stages: 0=Arithmetic, 1=Poly, 2=Trig, 3=Adv, 4=Complex
            CURRICULUM_LEVELS = [
                {'depth': 1, 'ops': ['+', '-', '*', '/']},
                {'depth': 2, 'ops': ['+', '-', '*', '/']},
                {'depth': 3, 'ops': ['+', '-', '*', '/', 'pow', 'sqrt']},
                {'depth': 4, 'ops': ['+', '-', '*', '/', 'pow', 'sqrt', 'sin', 'cos']},
                {'depth': 5, 'ops': None} # All
            ]
            
            # Initialize state if not present
            if 'curriculum_stage' not in locals():
                curriculum_stage = 0
            
            recent_rmse = np.mean(rmses[-20:]) if len(rmses) >= 20 else 1.0
            
            # Graduation condition: RMSE < 0.1 stable
            if len(rmses) > 20 and recent_rmse < 0.1 and curriculum_stage < len(CURRICULUM_LEVELS) - 1:
                curriculum_stage += 1
                stage_info = CURRICULUM_LEVELS[curriculum_stage]
                data_gen = DataGenerator(max_depth=stage_info['depth'], allowed_operators=stage_info['ops'])
                print(f"*** Curriculum Level Up! Stage {curriculum_stage} ({stage_info['depth']}, {stage_info['ops']}) ***")
                # Clear buffer to avoid training on old easy data? Maybe keep some for replay.
            
            # Ensure data_gen is initialized at start
            if iteration == 0:
                 stage_info = CURRICULUM_LEVELS[0]
                 data_gen = DataGenerator(max_depth=stage_info['depth'], allowed_operators=stage_info['ops'])

            stage_name = ["Arithmetic", "Polynomials", "Trigonometry", "Advanced", "Complex"][curriculum_stage]
            
            # Safe access to current_lr
            curr_lr_disp = optimizer.param_groups[0]['lr']
            msg = f"Iter {iteration+1}/{int(iterations)} [{stage_name}] RMSE:{recent_rmse:.3f} LR:{curr_lr_disp:.1e} | ETA: {eta_str}"
            progress((iteration + 1) / iterations, desc=msg)
            
            # Active Learning / Hard Mining Phase
            MODEL.eval()
            
            # Generate a large pool of candidates candidates to find the "hard" ones
            pool_size = problems_per_iter * 3  # Generate 3x more than we need
            candidates = data_gen.generate_inverse_batch(pool_size, point_count=int(point_count))
            
            if not candidates:
                continue
                
            # Quick forward pass to estimate difficulty (Loss)
            # We want to train on problems where the model currently FAILS (High Loss)
            hard_problems = []
            
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 32
                for i in range(0, len(candidates), chunk_size):
                    chunk = candidates[i:i+chunk_size]
                    
                    x_list = [d['x'] for d in chunk]
                    y_list = [d['y'] for d in chunk]
                    x_list, y_list = normalize_batch(x_list, y_list)
                    
                    token_lists = [[TOKEN_TO_ID.get(t, TOKEN_TO_ID['C']) for t in d['tokens']] for d in chunk]
                    max_len = max(len(s) for s in token_lists)
                    
                    # Prepare tensors
                    dec_in = torch.full((len(chunk), max_len + 1), SOS_ID, dtype=torch.long).to(DEVICE)
                    targets = torch.full((len(chunk), max_len + 1), -1, dtype=torch.long).to(DEVICE)
                    
                    for j, seq in enumerate(token_lists):
                        dec_in[j, 1:len(seq)+1] = torch.tensor(seq, dtype=torch.long)
                        targets[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                        
                    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
                    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
                    
                    logits, _ = MODEL(x_tensor, y_tensor, dec_in)
                    
                    # Calculate loss per item
                    # CrossEntropy usually aggregates, so we use reduction='none'
                    loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
                    raw_losses = loss_f(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
                    
                    # Reshape back to [Batch, Seq] to sum/mean per sample
                    raw_losses = raw_losses.view(len(chunk), -1)
                    # Average loss per non-padded token
                    mask = (targets != -1)
                    sample_losses = (raw_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
                    
                    for j, loss_val in enumerate(sample_losses):
                        # Store (Loss, Problem)
                        hard_problems.append((loss_val.item(), chunk[j]))
            
            # Sort by difficulty (Loss descending)
            hard_problems.sort(key=lambda x: x[0], reverse=True)
            
            # Stabilization: Mix Hardest (70%) + Random Examples (30%)
            # This prevents "Catastrophic Forgetting" of simpler patterns
            n_hard = int(problems_per_iter * 0.7)
            n_random = int(problems_per_iter) - n_hard
            
            # Top K hardest
            selected_hard = [p[1] for p in hard_problems[:n_hard]]
            
            # Random selection from the rest of the pool (to keep variety)
            remaining_pool = [p[1] for p in hard_problems[n_hard:]]
            selected_random = random.sample(remaining_pool, min(n_random, len(remaining_pool))) if remaining_pool else []
            
            selected_problems = selected_hard + selected_random
            
            avg_pool_loss = np.mean([p[0] for p in hard_problems])
            top_loss = np.mean([p[0] for p in hard_problems[:n_hard]]) if n_hard > 0 else 0
            
            print(f"Active Learning: Pool Loss {avg_pool_loss:.3f} -> Selected Mix (Hard:{top_loss:.3f})")

            # --- HALL OF SHAME CAPTURE ---
            # Capture what the model predicts for the top 3 hardest failures
            try:
                top_failures = hard_problems[:3]
                x_fail = [p[1]['x'].astype(np.float64) for p in top_failures]
                y_fail = [p[1]['y'].astype(np.float64) for p in top_failures]
                target_formulas = [p[1]['infix'] for p in top_failures]
                fail_losses = [p[0] for p in top_failures]
                
                # Simple Greedy Decode to see what it predicts
                from search.beam_search import BeamSearch
                # Use beam search with width 1 (Greedy) for speed, with curriculum mask
                bs = BeamSearch(MODEL, DEVICE, beam_width=1, max_length=20, curriculum_stage=curriculum_stage)
                
                for i in range(len(top_failures)):
                    try:
                        # Decode
                        # Enable return_partial to see what the model is thinking if it fails
                        res = bs.search(x_fail[i], y_fail[i], return_partial=True)
                        if not res:
                            pred_formula = "Search Empty (No Tokens)"
                        else:
                            pred_formula = res[0]['formula']
                        
                        add_training_error(
                            target=target_formulas[i],
                            predicted=pred_formula,
                            loss=fail_losses[i],
                            stage=stage_name
                        )
                    except Exception as e:
                        print(f"HoS Inner Error: {e}")
                        add_training_error(
                            target=target_formulas[i],
                            predicted=f"CRASH: {str(e)[:20]}",
                            loss=fail_losses[i],
                            stage=stage_name
                        )
            except Exception as e:
                import traceback
                print(f"HoS Outer Error: {e}")
                traceback.print_exc()

            # --- MCTS SOLVE ---
            for prob in selected_problems:
                x_data = prob['x'].astype(np.float64)
                y_data = prob['y'].astype(np.float64)
                
                try:
                    # Use MCTS to find the solution (or improve upon it)
                    # For inverse problems, we KNOW the solution, but MCTS helps explore variations
                    # and generates the policy distribution we want to learn.
                    result = searcher.search(x_data, y_data)
                    
                    # 1. Store Training Examples
                    if 'root' in result:
                        examples = searcher.get_training_examples(result['root'])
                        for (tokens, policy, value) in examples:
                            replay_buffer.append({
                                'x': x_data, 'y': y_data,
                                'tokens': tokens,
                                'policy': policy,
                                'value': value
                            })
                    
                    # 2. Track Metrics
                    if result.get('tokens'):
                        rmses.append(result['rmse'])
                        
                except Exception as e:
                    print(f"Self-play error: {e}")
                    continue
            
            # Training phase
            # To saturate GPU: Increase batch size and number of updates
            if len(replay_buffer) >= 64:
                MODEL.train()
                
                # Dynamic training steps: Train more if we have more data
                # AlphaZero ratio usually high (e.g. 10 epochs on new data)
                # Here we sample from buffer.
                train_batch_size = 128
                if len(replay_buffer) < train_batch_size:
                    train_batch_size = 64
                
                # Steps: roughly cover 20% of buffer or at least 10 steps
                steps = max(10, min(50, len(replay_buffer) // train_batch_size))
                
                for _ in range(steps):
                    batch = random.sample(list(replay_buffer), min(train_batch_size, len(replay_buffer)))
                    
                    x_list = [exp['x'] for exp in batch]
                    y_list = [exp['y'] for exp in batch]
                    x_list, y_list = normalize_batch(x_list, y_list)
                    
                    token_lists = [[TOKEN_TO_ID[t] for t in exp['tokens']] for exp in batch]
                    policy_targets = [exp['policy'] for exp in batch]
                    value_targets_list = [exp['value'] for exp in batch]
                    
                    max_len = max(len(s) for s in token_lists)
                    decoder_input = torch.full((len(batch), max_len + 1), SOS_ID, dtype=torch.long)
                    
                    # Policy targets (for KLDiv) and Value targets
                    policy_target_tensor = torch.tensor(np.array(policy_targets), dtype=torch.float32).to(DEVICE)
                    value_target_tensor = torch.tensor(np.array(value_targets_list), dtype=torch.float32).unsqueeze(1).to(DEVICE)
                    
                    for i, seq in enumerate(token_lists):
                        l = len(seq)
                        decoder_input[i, 1:l+1] = torch.tensor(seq, dtype=torch.long)
                    
                    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
                    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
                    decoder_input = decoder_input.to(DEVICE)
                    
                    optimizer.zero_grad()
                    logits, value_pred = MODEL(x_tensor, y_tensor, decoder_input)
                    
                    # Policy Loss (KL Divergence)
                    # Get logits for the last token position of each sequence
                    last_logits = []
                    for i, seq in enumerate(token_lists):
                        idx = len(seq) # Post-padding index? No, index in padded tensor.
                        # decoder_input: [SOS, T1, T2]
                        # logits: [PredSOS, PredT1, PredT2]
                        # We want prediction AFTER T2? No.
                        # MCTS Example: State=[T1, T2]. Policy=Dist for T3.
                        # Model Input: [SOS, T1, T2]. Output Last: Dist for T3.
                        # Index is len(seq).
                        last_logits.append(logits[i, idx, :VOCAB_SIZE])
                    
                    last_logits = torch.stack(last_logits)
                    log_probs = torch.nn.functional.log_softmax(last_logits, dim=1)
                    
                    loss_policy = kl_loss(log_probs, policy_target_tensor)
                    
                    # Value Loss (Quantile)
                    loss_value = quantile_loss_fn(value_pred, value_target_tensor)
                    
                    # Total Loss
                    loss = loss_policy + loss_value 
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
                        optimizer.step()
                        losses.append(loss.item())
            
            # Step Scheduler based on recent Loss
            if losses:
                current_loss = np.mean(losses[-10:])
                scheduler.step(current_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Periodic save
            if (iteration + 1) % 10 == 0:
                save_model()
        
        save_model()
        MODEL.eval()
        TRAINING_STATUS["running"] = False
        
        fig = create_selfplay_plot(losses, rmses)
        
        avg_rmse = np.mean(rmses[-50:]) if rmses else 0
        result = f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #ff6b6b;">
            <h2 style="color: #ff6b6b; margin: 0;">Self-Play Completado</h2>
            <p style="color: white;">Iteraciones: {int(iterations)} | Problemas: {len(rmses)}</p>
            <p style="color: #888;">RMSE Promedio: {avg_rmse:.4f} | Dispositivo: {DEVICE.type.upper()}</p>
        </div>
        """
        return result, fig
        
    except Exception as e:
        TRAINING_STATUS["running"] = False
        return f"Error: {str(e)}", None


def create_loss_plot(losses, title):
    """Create a loss plot with dark theme."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.plot(losses, color='#00d4ff', linewidth=2)
    ax.set_xlabel('Epoca', color='white')
    ax.set_ylabel('Loss', color='white')
    ax.set_title(title, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color('#00d4ff')
    plt.tight_layout()
    return fig


def create_selfplay_plot(losses, rmses):
    """Create dual plot for self-play results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='#1a1a2e')
    
    ax1.set_facecolor('#1a1a2e')
    if losses:
        ax1.plot(losses, color='#00d4ff', linewidth=2)
    ax1.set_xlabel('Step', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.set_title('Policy Loss', color='white', fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)
    
    ax2.set_facecolor('#1a1a2e')
    if rmses:
        ax2.plot(rmses, color='#ff6b6b', linewidth=1, alpha=0.5)
        if len(rmses) > 10:
            ma = np.convolve(rmses, np.ones(10)/10, mode='valid')
            ax2.plot(range(9, len(rmses)), ma, color='#ff6b6b', linewidth=2)
    ax2.set_xlabel('Problema', color='white')
    ax2.set_ylabel('RMSE', color='white')
    ax2.set_title('RMSE', color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2)
    
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#00d4ff')
    
    plt.tight_layout()
    return fig

def train_supervised(iterations, batch_size=128, point_count=10, progress=gr.Progress()):
    """
    Massive Supervised Pre-training (Warmup).
    Focus: Syntax, Basic Arithmetic, Overcoming "Collapse to Constant".
    Speed: High (No MCTS, just random generation + CrossEntropy).
    """
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    reset_stop_flag()  # Reset stop flag at start
    
    try:
        MODEL, DEVICE = get_model()
        
        MODEL.train()
        optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
        # Slower decay: T_max = iterations * 2 keeps LR higher for longer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(iterations*2), eta_min=1e-6)
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
        VOCAB_SIZE = len(VOCABULARY)
        SOS_ID = VOCAB_SIZE
        
        # Start extremely simple (Depth 1: x+1, x*x, etc.)
        allowed_ops = OPERATOR_STAGES[0]
        data_gen = DataGenerator(max_depth=1, allowed_operators=allowed_ops) 
        allowed_mask = get_allowed_token_mask(0, VOCAB_SIZE, DEVICE) # Stage 0 mask
        losses = []
        
        start_time = time.time()
        
        for i in range(int(iterations)):
            # Check for stop request
            if should_stop_training():
                print("⏹️ Pre-training stopped by user")
                break
            # ETA
            elapsed = time.time() - start_time
            if i > 0:
                iter_per_sec = i / elapsed
                remaining = int(iterations) - i
                eta = remaining / iter_per_sec
                eta_str = f"{eta:.0f}s"
            else:
                eta_str = "..."
                
            current_lr = optimizer.param_groups[0]['lr']
            msg = f"Iter {i+1}/{int(iterations)} Loss:{np.mean(losses[-50:]) if losses else 0:.3f} LR:{current_lr:.1e} ETA:{eta_str}"
            progress((i + 1) / iterations, desc=msg)
            
            # Generate Random Batch (High Speed)
            batch = data_gen.generate_batch(int(batch_size), point_count=int(point_count))
            
            if not batch:
                continue
            
            x_list = [d['x'] for d in batch]
            y_list = [d['y'] for d in batch]
            x_list, y_list = normalize_batch(x_list, y_list)
            
            token_lists = [[TOKEN_TO_ID.get(t, TOKEN_TO_ID['C']) for t in d['tokens']] for d in batch]
            
            max_len = max(len(s) for s in token_lists)
            decoder_input = torch.full((len(batch), max_len + 1), SOS_ID, dtype=torch.long)
            targets = torch.full((len(batch), max_len + 1), -1, dtype=torch.long)
            
            for j, seq in enumerate(token_lists):
                decoder_input[j, 1:len(seq)+1] = torch.tensor(seq, dtype=torch.long)
                targets[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                
            x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32).to(DEVICE)
            y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = MODEL(x_tensor, y_tensor, decoder_input)
            
            # Apply Stage 0 mask to bridge Pre-training with Curriculum
            # Use a more stable value (-1e4 instead of -1e9) to avoid overflow
            logits = logits + (1 - allowed_mask.view(1, 1, -1)) * -1e4
            
            loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                
            if (i+1) % 100 == 0:
                save_model()
                
        save_model()
        MODEL.eval()
        TRAINING_STATUS["running"] = False
        
        fig = create_loss_plot(losses, "Pre-Entrenamiento Supervisado")
        
        result = f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #ffd93d;">
            <h2 style="color: #ffd93d; margin: 0;">Escuela Primaria (Warmup) Completada</h2>
            <p style="color: white;">Iteraciones: {int(iterations)} | Loss Final: {losses[-1]:.4f}</p>
            <p style="color: #888;">El modelo ha aprendido sintaxis basica.</p>
        </div>
        """
        return result, fig
        
    except Exception as e:
        TRAINING_STATUS["running"] = False
        return f"Error: {str(e)}", None
