"""
Training functions for AlphaSymbolic Gradio App.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from collections import deque
import random

from grammar import VOCABULARY, TOKEN_TO_ID
from synthetic_data import DataGenerator
from app_core import get_model, save_model, TRAINING_STATUS


def train_basic(epochs, batch_size, progress=gr.Progress()):
    """Basic training with synthetic data."""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "⚠️ Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    MODEL, DEVICE = get_model()
    
    MODEL.train()
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    VOCAB_SIZE = len(VOCABULARY)
    SOS_ID = VOCAB_SIZE
    
    data_gen = DataGenerator(max_depth=4)
    losses = []
    
    for epoch in range(int(epochs)):
        progress((epoch + 1) / epochs, desc=f"Época {epoch+1}/{int(epochs)} [{DEVICE.type.upper()}]")
        
        batch = data_gen.generate_batch(int(batch_size))
        if len(batch) < 2:
            continue
        
        x_list = [d['x'] for d in batch]
        y_list = [d['y'] for d in batch]
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
        
        logits, _ = MODEL(x_tensor, y_tensor, decoder_input)
        loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    save_model()
    MODEL.eval()
    TRAINING_STATUS["running"] = False
    
    fig = create_loss_plot(losses, "📉 Entrenamiento Básico")
    
    result = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #4ade80;">
        <h2 style="color: #4ade80; margin: 0;">✅ Entrenamiento Completado</h2>
        <p style="color: white;">Épocas: {int(epochs)} | Loss Final: {losses[-1]:.4f}</p>
        <p style="color: #00d4ff;">Dispositivo: {DEVICE.type.upper()}</p>
    </div>
    """
    return result, fig


def train_curriculum(epochs, batch_size, progress=gr.Progress()):
    """Curriculum Learning - starts simple, increases difficulty."""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "⚠️ Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    MODEL, DEVICE = get_model()
    
    MODEL.train()
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    VOCAB_SIZE = len(VOCABULARY)
    SOS_ID = VOCAB_SIZE
    losses = []
    
    for epoch in range(int(epochs)):
        curriculum_progress = min(epoch / (epochs * 0.7), 1.0)
        current_depth = int(2 + curriculum_progress * 4)
        
        progress((epoch + 1) / epochs, desc=f"Época {epoch+1}/{int(epochs)} (prof: {current_depth}) [{DEVICE.type.upper()}]")
        
        data_gen = DataGenerator(max_depth=current_depth)
        batch = data_gen.generate_batch(int(batch_size))
        if len(batch) < 2:
            continue
        
        x_list = [d['x'] for d in batch]
        y_list = [d['y'] for d in batch]
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
        
        logits, _ = MODEL(x_tensor, y_tensor, decoder_input)
        loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    save_model()
    MODEL.eval()
    TRAINING_STATUS["running"] = False
    
    fig = create_loss_plot(losses, "📉 Curriculum Learning")
    
    result = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #00d4ff;">
        <h2 style="color: #00d4ff; margin: 0;">🎓 Curriculum Learning Completado</h2>
        <p style="color: white;">Épocas: {int(epochs)} | Loss Final: {losses[-1]:.4f}</p>
        <p style="color: #888;">Profundidad máxima: 6 | Dispositivo: {DEVICE.type.upper()}</p>
    </div>
    """
    return result, fig


def train_self_play(iterations, problems_per_iter, progress=gr.Progress()):
    """AlphaZero Self-Play loop."""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["running"]:
        return "⚠️ Entrenamiento ya en progreso", None
    
    TRAINING_STATUS["running"] = True
    MODEL, DEVICE = get_model()
    
    from beam_search import BeamSearch
    
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=1e-4, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    VOCAB_SIZE = len(VOCABULARY)
    SOS_ID = VOCAB_SIZE
    
    replay_buffer = deque(maxlen=10000)
    data_gen = DataGenerator(max_depth=5)
    searcher = BeamSearch(MODEL, DEVICE, beam_width=8, max_length=20)
    
    rmses = []
    losses = []
    
    for iteration in range(int(iterations)):
        progress((iteration + 1) / iterations, desc=f"Iter {iteration+1}/{int(iterations)} [{DEVICE.type.upper()}]")
        
        # Self-play phase
        MODEL.eval()
        problems = data_gen.generate_batch(int(problems_per_iter))
        
        for prob in problems:
            x_data = prob['x'].astype(np.float64)
            y_data = prob['y'].astype(np.float64)
            
            results = searcher.search(x_data, y_data)
            if results:
                best = results[0]
                replay_buffer.append({
                    'x': x_data, 'y': y_data,
                    'tokens': best['tokens'],
                    'rmse': best['rmse']
                })
                rmses.append(best['rmse'])
        
        # Training phase
        if len(replay_buffer) >= 16:
            MODEL.train()
            batch = random.sample(list(replay_buffer), min(32, len(replay_buffer)))
            
            x_list = [exp['x'] for exp in batch]
            y_list = [exp['y'] for exp in batch]
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
            
            logits, _ = MODEL(x_tensor, y_tensor, decoder_input)
            loss = ce_loss(logits.view(-1, VOCAB_SIZE + 1), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
    
    save_model()
    MODEL.eval()
    TRAINING_STATUS["running"] = False
    
    fig = create_selfplay_plot(losses, rmses)
    
    avg_rmse = np.mean(rmses[-50:]) if rmses else 0
    result = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #ff6b6b;">
        <h2 style="color: #ff6b6b; margin: 0;">🧠 Self-Play Completado</h2>
        <p style="color: white;">Iteraciones: {int(iterations)} | Problemas: {len(rmses)}</p>
        <p style="color: #888;">RMSE Promedio: {avg_rmse:.4f} | Dispositivo: {DEVICE.type.upper()}</p>
    </div>
    """
    return result, fig


def create_loss_plot(losses, title):
    """Create a loss plot with dark theme."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.plot(losses, color='#00d4ff', linewidth=2)
    ax.set_xlabel('Época', color='white')
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
    ax1.plot(losses, color='#00d4ff', linewidth=2)
    ax1.set_xlabel('Step', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.set_title('📉 Policy Loss', color='white', fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)
    
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(rmses, color='#ff6b6b', linewidth=1, alpha=0.5)
    if len(rmses) > 10:
        ma = np.convolve(rmses, np.ones(10)/10, mode='valid')
        ax2.plot(range(9, len(rmses)), ma, color='#ff6b6b', linewidth=2)
    ax2.set_xlabel('Problema', color='white')
    ax2.set_ylabel('RMSE', color='white')
    ax2.set_title('📊 RMSE', color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2)
    
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('#00d4ff')
    
    plt.tight_layout()
    return fig
