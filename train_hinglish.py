import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from gnn_rwkv_story_gen import SocialNarrativeModel, get_data

def train_hinglish():
    print("--- Training Hinglish Social Brain (10 Epochs) ---")
    
    # 1. Load Data
    data, w2i, i2w = get_data("hinglish_train.txt")
    vocab_size = len(w2i)
    num_nodes = 5 # Standard graph size
    dim = 64
    
    # Save Vocab for future loading
    with open("vocab.json", "w", encoding='utf-8') as f:
        json.dump({"w2i": w2i, "i2w": {str(k): v for k, v in i2w.items()}}, f)
    
    # 2. Initialize Model
    model = SocialNarrativeModel(vocab_size, num_nodes, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Standard social graph context
    edges = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 1]], dtype=torch.long)
    node_ids = torch.tensor([0, 1, 2, 3, 4])
    
    # 3. Training Loop
    model.train()
    for epoch in range(10): # 10 Epochs as requested
        h = torch.zeros((num_nodes, dim))
        optimizer.zero_grad()
        
        input_seq = torch.tensor(data[:-1])
        target_seq = torch.tensor(data[1:])
        
        logits, _ = model(input_seq, node_ids, edges, h)
        loss = F.cross_entropy(logits, target_seq)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")
    
    # 4. Save Weights
    torch.save(model.state_dict(), "social_brain.pth")
    print("--- Training Complete. Weights saved to social_brain.pth ---")

if __name__ == "__main__":
    train_hinglish()
