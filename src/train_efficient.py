"""
Efficient Training Module - SALT + TLT methodology
Based on:
1. SALT: Small model Aided Large model Training (Google Research)
2. TLT: Taming the Long Tail (MIT) - Adaptive Speculative Decoding

Key optimizations:
- Knowledge Distillation from smaller teacher model
- Data selection based on example difficulty
- Adaptive node activation for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.gnn_rwkv_story_gen import (
    MODEL_CONTEXT,
    MODEL_DIM,
    MODEL_LAYERS,
    MODEL_NODES,
    SocialNarrativeModel,
    TOKENIZER_VERSION,
    get_data_from_text,
    AdaptiveSocialBrain,
)


class EfficientTrainer:
    """
    SALT + TLT inspired efficient training:
    1. Stage 1: Knowledge Distillation from smaller model (teacher)
    2. Stage 2: Standard pretraining with data selection
    3. Adaptive node routing based on example difficulty
    """
    
    def __init__(
        self,
        vocab_size,
        num_nodes=MODEL_NODES,
        dim=MODEL_DIM,
        num_layers=MODEL_LAYERS,
        max_seq=MODEL_CONTEXT,
        teacher_dim=None,
        teacher_nodes=None,
        use_kd=True,
        kd_temperature=2.0,
        use_data_selection=True,
        selection_top_k=10,
    ):
        self.vocab_size = vocab_size
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq = max_seq
        self.use_kd = use_kd
        self.kd_temperature = kd_temperature
        self.use_data_selection = use_data_selection
        self.selection_top_k = selection_top_k
        
        # Create smaller teacher model for KD
        if use_kd and teacher_dim is None:
            teacher_dim = max(64, dim // 2)
            teacher_nodes = max(8, num_nodes // 2)
        
        if use_kd:
            self.teacher = SocialNarrativeModel(
                vocab_size, teacher_nodes, dim=dim, 
                num_layers=max(1, num_layers // 2), max_seq=max_seq
            )
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            print(f"[SALT] Teacher model: {teacher_dim}d, {teacher_nodes} nodes, {num_layers//2} layers")
        
        # Main student model
        self.model = SocialNarrativeModel(
            vocab_size, num_nodes, dim=dim, 
            num_layers=num_layers, max_seq=max_seq
        )
        
        # Example difficulty tracker
        self.example_losses = defaultdict(float)
        self.example_counts = defaultdict(int)
        
    def compute_kd_loss(self, student_logits, teacher_logits, target):
        """Knowledge Distillation loss with temperature scaling"""
        T = self.kd_temperature
        
        # Student soft labels
        student_soft = F.log_softmax(student_logits.view(-1, self.vocab_size) / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits.view(-1, self.vocab_size) / T, dim=-1)
        
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        # Also add standard CE loss
        ce_loss = F.cross_entropy(student_logits.view(-1, self.vocab_size), target.view(-1))
        
        return kd_loss + 0.5 * ce_loss
    
    def select_hard_examples(self, input_ids, model, device, top_k=10):
        """
        Select hard examples using SALT methodology:
        - Compute per-token losses
        - Select examples where teacher performs well but not perfectly
        - Focus on challenging yet learnable examples
        """
        if not self.use_data_selection:
            return list(range(len(input_ids)))
        
        model.eval()
        with torch.no_grad():
            chunk_size = 32
            example_scores = []
            
            for i in range(0, len(input_ids) - chunk_size, chunk_size):
                chunk = torch.tensor(input_ids[i:i+chunk_size], device=device)
                target = torch.tensor(input_ids[i+1:i+chunk_size+1], device=device)
                node_ids = torch.arange(self.num_nodes, device=device)
                h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
                
                logits, h = model(chunk, node_ids, None, h)
                
                # Compute per-token loss
                log_probs = F.log_softmax(logits.view(-1, self.vocab_size), dim=-1)
                target_one_hot = F.one_hot(target, num_classes=self.vocab_size).float()
                token_loss = -(target_one_hot * log_probs).sum(dim=-1)
                
                # Score: higher = harder but learnable
                # Skip examples where model is too good (low loss = easy)
                # Skip examples where model fails completely (loss too high = impossible)
                score = token_loss.mean().item()
                example_scores.append((i, score))
            
            # Sort by difficulty, select middle-range (hard but learnable)
            example_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top_k hardest but still in learnable range
            selected_indices = []
            for idx, score in example_scores[:top_k * 2]:
                if 0.5 < score < 5.0:  # Learnable range
                    selected_indices.append(idx)
                if len(selected_indices) >= top_k:
                    break
            
            # If not enough selected, add more from middle
            if len(selected_indices) < top_k // 2:
                for idx, score in example_scores[top_k:top_k*3]:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
            
            return selected_indices if selected_scores else list(range(min(len(input_ids), top_k * 2)))
    
    def train(
        self,
        input_ids,
        epochs=50,
        device='cpu',
        kd_epochs=10,
        verbose=True,
        progress_callback=None,
    ):
        """Two-stage training: KD first, then standard"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        node_ids = torch.arange(self.num_nodes, device=device)
        
        total_batches = max(1, (len(input_ids) - 32) // 32)
        
        # ===== STAGE 1: Knowledge Distillation (SALT) =====
        if self.use_kd and kd_epochs > 0:
            if verbose:
                print(f"[SALT] Stage 1: Knowledge Distillation ({kd_epochs} epochs)")
            
            for epoch in range(kd_epochs):
                self.model.train()
                total_loss = 0
                h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
                
                # Select hard examples for KD
                hard_indices = self.select_hard_examples(
                    input_ids, self.model, device, top_k=self.selection_top_k
                )
                
                for i in hard_indices:
                    if i + 32 >= len(input_ids) - 1:
                        continue
                    
                    chunk = torch.tensor(input_ids[i:i+32], device=device)
                    target = torch.tensor(input_ids[i+1:i+33], device=device)
                    
                    # Get teacher predictions
                    with torch.no_grad():
                        teacher_logits, _ = self.teacher(chunk, node_ids, None, 
                            [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.teacher.num_layers)])
                    
                    # Student forward
                    optimizer.zero_grad()
                    student_logits, h = self.model(chunk, node_ids, None, h)
                    h = [s.detach() for s in h]
                    
                    # KD loss
                    loss = self.compute_kd_loss(student_logits, teacher_logits.detach(), target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if verbose and (epoch + 1) % 2 == 0:
                    avg_loss = total_loss / max(1, len(hard_indices))
                    print(f"  KD Epoch {epoch+1}/{kd_epochs} | Loss: {avg_loss:.4f}")
        
        # ===== STAGE 2: Standard Pretraining with Data Selection =====
        if verbose:
            print(f"[SALT] Stage 2: Standard Pre-training ({epochs - kd_epochs} epochs)")
        
        for epoch in range(kd_epochs, epochs):
            self.model.train()
            total_loss = 0
            h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
            
            # Select hard examples for focused training
            if self.use_data_selection and epoch % 3 == 0:
                hard_indices = self.select_hard_examples(
                    input_ids, self.model, device, top_k=self.selection_top_k
                )
            else:
                hard_indices = list(range(0, len(input_ids) - 32, 32))
            
            steps = 0
            for i in hard_indices:
                if i + 32 >= len(input_ids) - 1:
                    continue
                
                chunk = torch.tensor(input_ids[i:i+32], device=device)
                target = torch.tensor(input_ids[i+1:i+33], device=device)
                
                optimizer.zero_grad()
                logits, h = self.model(chunk, node_ids, None, h)
                h = [s.detach() for s in h]
                
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(1, steps)
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
                if progress_callback:
                    progress_callback(epoch + 1, epochs, avg_loss)
        
        return self.model


class AdaptiveNodeRouter:
    """
    TLT-inspired adaptive node routing:
    - Uses LIF neurons efficiently
    - Focuses on "important" nodes for hard examples
    - Reduces computation on easy inputs
    """
    
    def __init__(self, num_nodes, max_active):
        self.num_nodes = num_nodes
        self.max_active = max(1, min(max_active, num_nodes))
        self.node_importance = torch.ones(num_nodes)
        
    def compute_importance(self, node_activations):
        """Track which nodes are most important"""
        with torch.no_grad():
            importance = node_activations.abs().mean(dim=-1)
            self.node_importance = 0.9 * self.node_importance + 0.1 * importance
    
    def get_active_mask(self, difficulty_score):
        """
        Adaptive node activation based on input difficulty:
        - Hard inputs: activate more nodes
        - Easy inputs: activate fewer nodes (efficiency)
        """
        # Map difficulty (0-10) to node count
        base_nodes = self.max_active // 2
        extra_nodes = int((difficulty_score / 10.0) * (self.max_active - base_nodes))
        active_count = min(self.max_active, base_nodes + extra_nodes)
        
        # Select top-k important nodes
        _, top_indices = torch.topk(self.node_importance, active_count)
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[top_indices] = True
        
        return mask


def train_efficient(
    input_file,
    epochs=30,
    kd_epochs=8,
    persist=True,
    max_sentences=None,
    verbose=True,
    use_kd=True,
    use_data_selection=True,
):
    """
    Efficient training using SALT methodology:
    1. First kd_epochs: Knowledge Distillation from smaller model
    2. Remaining epochs: Standard pretraining with data selection
    """
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return None
    
    # Load and tokenize data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read().lower().strip()
    
    if max_sentences:
        lines = [l.strip() for l in text.split('\n') if l.strip()][:max_sentences]
        text = '\n'.join(lines)
    
    data, w2i, i2w = get_data_from_text(text)
    vocab_size = len(w2i)
    
    if verbose:
        print(f"[Efficient Train] Vocab: {vocab_size}, Data points: {len(data)}")
        print(f"[Config] KD: {use_kd}, Data Selection: {use_data_selection}, KD epochs: {kd_epochs}/{epochs}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize trainer
    trainer = EfficientTrainer(
        vocab_size=vocab_size,
        num_nodes=MODEL_NODES,
        dim=MODEL_DIM,
        num_layers=MODEL_LAYERS,
        max_seq=MODEL_CONTEXT,
        use_kd=use_kd,
        use_data_selection=use_data_selection,
    )
    
    if device != 'cpu':
        trainer.model = trainer.model.to(device)
        if trainer.teacher:
            trainer.teacher = trainer.teacher.to(device)
    
    # Train
    model = trainer.train(
        input_ids=data,
        epochs=epochs,
        device=device,
        kd_epochs=kd_epochs,
        verbose=verbose,
    )
    
    if verbose:
        print("[Efficient Train] Training complete!")
    
    return {
        'model': model,
        'w2i': w2i,
        'i2w': i2w,
        'vocab_size': vocab_size,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficient SALT Training")
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--kd-epochs", type=int, default=8, help="Epochs for KD phase")
    parser.add_argument("--no-kd", action="store_true", help="Disable Knowledge Distillation")
    parser.add_argument("--no-selection", action="store_true", help="Disable data selection")
    parser.add_argument("--max-sentences", type=int, default=None)
    
    args = parser.parse_args()
    
    result = train_efficient(
        args.file,
        epochs=args.epochs,
        kd_epochs=args.kd_epochs,
        use_kd=not args.no_kd,
        use_data_selection=not args.no_selection,
        max_sentences=args.max_sentences,
    )
    
    if result:
        print(f"Training complete! Vocab size: {result['vocab_size']}")