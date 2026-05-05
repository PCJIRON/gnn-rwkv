import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import time
from gnn_rwkv_story_gen import (
    SocialNarrativeModel,
    build_vocab,
    normalize_text,
    MODEL_CONTEXT,
    MODEL_DIM
)

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

WEIGHTS_PATH = "social_brain.pth"
VOCAB_PATH = "vocab.json"

def train_generative_brain(input_file, epochs=50, batch_size=4):
    if not os.path.exists(input_file): return
    
    with open(input_file, "r", encoding='utf-8') as f:
        text = f.read()
    
    w2i, i2w = build_vocab(text)
    vocab_size = len(w2i)
    
    # Tokenize full text
    tokens = [w2i[w] for w in re.findall(r"\w+|[^\w\s]", normalize_text(text)) if w in w2i]
    
    model = SocialNarrativeModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    print(f"--- Training Generative Brain v4.0: {epochs} epochs, Vocab: {vocab_size} ---")
    
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        # Simple batching
        chunk_size = 64
        num_steps = len(tokens) // (batch_size * chunk_size)
        
        for i in range(num_steps):
            batch_x = []
            batch_y = []
            for b in range(batch_size):
                idx = i * chunk_size + (b * (len(tokens) // batch_size))
                batch_x.append(torch.tensor(tokens[idx : idx + chunk_size]))
                batch_y.append(torch.tensor(tokens[idx + 1 : idx + chunk_size + 1]))
            
            x = torch.stack(batch_x)
            y = torch.stack(batch_y)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/max(1, num_steps):.4f} | Time: {time.time()-start_time:.2f}s")
        
    # Save
    torch.save(model.state_dict(), WEIGHTS_PATH)
    with open(VOCAB_PATH, "w", encoding='utf-8') as f:
        json.dump({"w2i": w2i, "i2w": i2w}, f)
    print("--- Brain Saved ---")

import re
if __name__ == "__main__":
    train_generative_brain("wiki_data.txt", epochs=50)
