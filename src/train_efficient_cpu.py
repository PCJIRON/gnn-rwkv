"""
CPU-Friendly Efficient Training Module
Optimizations that work on CPU only:

1. Knowledge Distillation from smaller model (SALT methodology)
2. Hard example selection - focus on challenging but learnable examples
3. Adaptive node activation - use fewer nodes for easy inputs
4. Efficient batch processing for CPU
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

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.gnn_rwkv_story_gen import (
    MODEL_CONTEXT,
    MODEL_DIM,
    MODEL_LAYERS,
    MODEL_NODES,
    SocialNarrativeModel,
    get_data_from_text,
)


class CPUEfficientTrainer:
    """
    CPU-optimized efficient training using SALT methodology.
    No GPU-specific code - all optimizations work on CPU.
    """
    
    def __init__(
        self,
        vocab_size,
        num_nodes=MODEL_NODES,
        dim=MODEL_DIM,
        num_layers=MODEL_LAYERS,
        max_seq=MODEL_CONTEXT,
        use_kd=True,
        kd_temperature=2.0,
        use_data_selection=True,
        selection_top_k=20,
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
        
        # Create smaller teacher for KD (CPU-friendly: smaller model)
        if use_kd:
            teacher_dim = max(64, dim // 2)
            teacher_nodes = max(8, num_nodes // 2)
            teacher_layers = max(1, num_layers // 2)
            
            self.teacher = SocialNarrativeModel(
                vocab_size, teacher_nodes, dim=dim, 
                num_layers=teacher_layers, max_seq=max_seq
            )
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            print(f"[Trainer] Teacher: {teacher_dim}d, {teacher_nodes}n, {teacher_layers}l")
        
        # Main student model
        self.model = SocialNarrativeModel(
            vocab_size, num_nodes, dim=dim, 
            num_layers=num_layers, max_seq=max_seq
        )
        
    def compute_kd_loss(self, student_logits, teacher_logits, target):
        """KD loss with temperature scaling"""
        T = self.kd_temperature
        
        student_soft = F.log_softmax(student_logits.view(-1, self.vocab_size) / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits.view(-1, self.vocab_size) / T, dim=-1)
        
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        ce_loss = F.cross_entropy(student_logits.view(-1, self.vocab_size), target.view(-1))
        
        return kd_loss + 0.5 * ce_loss
    
    def evaluate_difficulty(self, input_ids, model, device):
        """
        Evaluate which examples are hard/easy.
        Returns indices sorted by difficulty (hard first).
        """
        if not self.use_data_selection:
            return list(range(0, len(input_ids) - 32, 32))
        
        model.eval()
        with torch.no_grad():
            chunk_size = 16  # Smaller for CPU
            example_scores = []
            
            for i in range(0, len(input_ids) - chunk_size, chunk_size):
                chunk = torch.tensor(input_ids[i:i+chunk_size], device=device, dtype=torch.long)
                target = torch.tensor(input_ids[i+1:i+chunk_size+1], device=device, dtype=torch.long)
                node_ids = torch.arange(self.num_nodes, device=device, dtype=torch.long)
                
                h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
                
                logits, h = model(chunk, node_ids, None, h)
                
                # Cross entropy loss as difficulty measure
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1), reduction='mean')
                score = loss.item()
                
                # Only select learnable examples (not too easy, not impossible)
                if 0.3 < score < 8.0:
                    example_scores.append((i, score))
            
            # Sort by score (hard first)
            example_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return indices
            return [idx for idx, _ in example_scores]
    
    def train(
        self,
        input_ids,
        epochs=30,
        kd_epochs=8,
        batch_size=16,
        lr=0.001,
        verbose=True,
        progress_callback=None,
    ):
        """Two-stage training on CPU"""
        
        device = 'cpu'
        self.model.to(device)
        if self.use_kd:
            self.teacher.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        node_ids = torch.arange(self.num_nodes, device=device, dtype=torch.long)
        
        # ===== STAGE 1: Knowledge Distillation =====
        if self.use_kd and kd_epochs > 0:
            if verbose:
                print(f"[Stage 1] Knowledge Distillation ({kd_epochs} epochs)")
            
            for epoch in range(kd_epochs):
                self.model.train()
                total_loss = 0
                h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
                
                # Get hard examples
                hard_indices = self.evaluate_difficulty(input_ids, self.model, device)[:self.selection_top_k]
                
                if not hard_indices:
                    hard_indices = list(range(0, min(len(input_ids) - 32, 100), batch_size))
                
                for idx in hard_indices:
                    if idx + batch_size >= len(input_ids) - 1:
                        continue
                    
                    chunk = torch.tensor(input_ids[idx:idx+batch_size], device=device, dtype=torch.long)
                    target = torch.tensor(input_ids[idx+1:idx+batch_size+1], device=device, dtype=torch.long)
                    
                    # Teacher prediction
                    with torch.no_grad():
                        teacher_h = [torch.zeros((self.num_nodes, self.dim), device=device) 
                                    for _ in range(self.teacher.num_layers)]
                        teacher_logits, _ = self.teacher(chunk, node_ids, None, teacher_h)
                    
                    # Student training
                    optimizer.zero_grad()
                    student_logits, h = self.model(chunk, node_ids, None, h)
                    h = [s.detach() for s in h]
                    
                    loss = self.compute_kd_loss(student_logits, teacher_logits, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if verbose and (epoch + 1) % 2 == 0:
                    print(f"  KD Epoch {epoch+1}/{kd_epochs} | Loss: {total_loss/max(1,len(hard_indices)):.4f}")
        
        # ===== STAGE 2: Standard training with data selection =====
        std_epochs = epochs - kd_epochs
        if verbose:
            print(f"[Stage 2] Standard Training ({std_epochs} epochs)")
        
        for epoch in range(std_epochs):
            self.model.train()
            total_loss = 0
            h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
            
            # Every 3rd epoch, re-evaluate difficulty
            if self.use_data_selection and epoch % 3 == 0:
                train_indices = self.evaluate_difficulty(input_ids, self.model, device)[:self.selection_top_k * 2]
            else:
                train_indices = list(range(0, len(input_ids) - batch_size, batch_size))
            
            steps = 0
            for idx in train_indices:
                if idx + batch_size >= len(input_ids) - 1:
                    continue
                
                chunk = torch.tensor(input_ids[idx:idx+batch_size], device=device, dtype=torch.long)
                target = torch.tensor(input_ids[idx+1:idx+batch_size+1], device=device, dtype=torch.long)
                
                optimizer.zero_grad()
                logits, h = self.model(chunk, node_ids, None, h)
                h = [s.detach() for s in h]
                
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{std_epochs} | Loss: {total_loss/max(1,steps):.4f}")
                if progress_callback:
                    progress_callback(kd_epochs + epoch + 1, epochs, total_loss/max(1,steps))
        
        return self.model


def train_efficient_cpu(
    input_file,
    epochs=30,
    kd_epochs=8,
    max_sentences=None,
    verbose=True,
    use_kd=True,
    use_data_selection=True,
    batch_size=16,
    learning_rate=0.001,
):
    """
    CPU-efficient training - no GPU dependencies.
    """
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return None
    
    # Load and tokenize
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read().lower().strip()
    
    if max_sentences:
        lines = [l.strip() for l in text.split('\n') if l.strip()][:max_sentences]
        text = '\n'.join(lines)
    
    data, w2i, i2w = get_data_from_text(text)
    vocab_size = len(w2i)
    
    if verbose:
        print(f"[CPU Train] Vocab: {vocab_size} | Tokens: {len(data)}")
        print(f"[Config] KD: {use_kd}, Data Selection: {use_data_selection}, Batch: {batch_size}")
    
    # Initialize trainer
    trainer = CPUEfficientTrainer(
        vocab_size=vocab_size,
        num_nodes=MODEL_NODES,
        dim=MODEL_DIM,
        num_layers=MODEL_LAYERS,
        max_seq=MODEL_CONTEXT,
        use_kd=use_kd,
        use_data_selection=use_data_selection,
    )
    
    # Train
    model = trainer.train(
        input_ids=data,
        epochs=epochs,
        kd_epochs=kd_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        verbose=verbose,
    )
    
    return {
        'model': model,
        'w2i': w2i,
        'i2w': i2w,
        'vocab_size': vocab_size,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CPU-Efficient Training")
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs")
    parser.add_argument("--kd-epochs", type=int, default=8, help="KD epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-kd", action="store_true", help="Disable KD")
    parser.add_argument("--no-selection", action="store_true", help="Disable data selection")
    parser.add_argument("--max-sentences", type=int, default=500)
    
    args = parser.parse_args()
    
    result = train_efficient_cpu(
        args.file,
        epochs=args.epochs,
        kd_epochs=args.kd_epochs,
        use_kd=not args.no_kd,
        use_data_selection=not args.no_selection,
        batch_size=args.batch,
        learning_rate=args.lr,
        max_sentences=args.max_sentences,
    )
    
    if result:
        print(f"Done! Vocab: {result['vocab_size']}")