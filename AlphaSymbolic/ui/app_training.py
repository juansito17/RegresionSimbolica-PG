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

from core.grammar import VOCABULARY, TOKEN_TO_ID
from data.synthetic_data import DataGenerator
from ui.app_core import get_model, save_model, TRAINING_STATUS


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
            logits, _ = MODEL(x_tensor, y_tensor, decoder_input)
            loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
            
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
    
    try:
        MODEL, DEVICE = get_model()
        
        from search.mcts import MCTS
        
        optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
        VOCAB_SIZE = len(VOCABULARY)
        SOS_ID = VOCAB_SIZE
        
        replay_buffer = deque(maxlen=10000)
        data_gen = DataGenerator(max_depth=5)
        # Use MCTS for self-play as per AlphaZero
        searcher = MCTS(MODEL, DEVICE, max_simulations=50)
        
        rmses = []
        losses = []
        
        for iteration in range(int(iterations)):
            progress((iteration + 1) / iterations, desc=f"Iter {iteration+1}/{int(iterations)} [{DEVICE.type.upper()}]")
            
            # Self-play phase
            MODEL.eval()
            
            # Generate mix of problems: 50% inverse (solvable), 50% random
            n_inverse = int(problems_per_iter) // 2
            n_random = int(problems_per_iter) - n_inverse
            
            probs_inv = data_gen.generate_inverse_batch(n_inverse, point_count=int(point_count))
            probs_rnd = data_gen.generate_batch(n_random, point_count=int(point_count))
            problems = probs_inv + probs_rnd
            
            for prob in problems:
                x_data = prob['x'].astype(np.float64)
                y_data = prob['y'].astype(np.float64)
                
                try:
                    result = searcher.search(x_data, y_data)
                    if result and result.get('tokens'):
                        replay_buffer.append({
                            'x': x_data, 'y': y_data,
                            'tokens': result['tokens'],
                            'rmse': result['rmse']
                        })
                        rmses.append(result['rmse'])
                except Exception as e:
                    print(f"Self-play error: {e}")
                    continue
            
            # Training phase
            if len(replay_buffer) >= 16:
                MODEL.train()
                batch = random.sample(list(replay_buffer), min(32, len(replay_buffer)))
                
                x_list = [exp['x'] for exp in batch]
                y_list = [exp['y'] for exp in batch]
                x_list, y_list = normalize_batch(x_list, y_list)
                
                token_lists = [[TOKEN_TO_ID[t] for t in exp['tokens']] for exp in batch]
                
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
                
                # Prepare value targets based on RMSE
                # Transform RMSE -> Value [0, 1] (1 = perfect match)
                rmses_batch = [exp['rmse'] for exp in batch]
                value_targets = torch.tensor([1.0 / (1.0 + r) for r in rmses_batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                
                optimizer.zero_grad()
                logits, value_pred = MODEL(x_tensor, y_tensor, decoder_input)
                
                # Policy Loss (Cross Entropy)
                loss_policy = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
                
                # Value Loss (MSE)
                # We want value_pred to match the "quality" of the formula associated with this state
                # Note: value_pred corresponds to the LAST token in sequence
                value_pred_last = value_pred  # It's already [batch, 1] from just the last token in model.py
                loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
                
                # Total Loss = Policy + Value
                loss = loss_policy + loss_value
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())
            
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
