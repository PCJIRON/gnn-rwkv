import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import argparse
from gnn_rwkv_story_gen import SocialNarrativeModel, get_data

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def incremental_train(input_file, epochs=10):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    weights_path = "social_brain.pth"
    vocab_path = "vocab.json"
    dim = 64
    num_nodes = 5
    
    # 1. Load Existing Knowledge
    existing_w2i = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding='utf-8') as f:
            v_data = json.load(f)
            existing_w2i = v_data["w2i"]
            print(f"Found existing vocabulary: {len(existing_w2i)} words.")
    
    # 2. Get New Data and Identify New Words
    new_data_raw, new_w2i, new_i2w = get_data(input_file)
    
    # Merging Logic: Only add words that are NOT in existing_w2i
    merged_w2i = existing_w2i.copy()
    new_word_count = 0
    for word in new_w2i:
        if word not in merged_w2i:
            merged_w2i[word] = len(merged_w2i)
            new_word_count += 1
    
    merged_i2w = {i: w for w, i in merged_w2i.items()}
    vocab_size = len(merged_w2i)
    print(f"Merged Vocabulary: {vocab_size} words ({new_word_count} new words added).")
    
    # Convert new_data to merged indices
    data = [merged_w2i[new_i2w[idx]] for idx in new_data_raw]
    
    # 3. Initialize/Expand Model
    model = SocialNarrativeModel(vocab_size, num_nodes, dim)
    
    if os.path.exists(weights_path):
        print("Loading and expanding existing brain weights...")
        old_state = torch.load(weights_path)
        
        # Smart Weight Copying for Embeddings (Resizing)
        new_state = model.state_dict()
        for key in old_state:
            if key in new_state:
                old_weight = old_state[key]
                new_weight = new_state[key]
                
                if old_weight.shape == new_weight.shape:
                    new_state[key] = old_weight
                else:
                    # Partial copy for resized layers (Embedding and Linear Head)
                    print(f"Resizing layer: {key} from {old_weight.shape} to {new_weight.shape}")
                    if len(old_weight.shape) == 2:
                        new_state[key][:old_weight.shape[0], :old_weight.shape[1]] = old_weight
                    else:
                        new_state[key][:old_weight.shape[0]] = old_weight
        
        model.load_state_dict(new_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    edges = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 1]], dtype=torch.long)
    node_ids = torch.tensor([0, 1, 2, 3, 4])
    
    # 4. Training Loop (Relationship Update)
    print(f"Updating relationships for {epochs} epochs...")
    model.train()
    chunk_size = 500 # Smaller chunks for memory efficiency
    
    for epoch in range(epochs):
        epoch_loss = 0
        chunks_count = 0
        
        # Divide data into chunks
        for i in range(0, len(data) - chunk_size, chunk_size):
            h = torch.zeros((num_nodes, dim)) # Reset state for each chunk
            optimizer.zero_grad()
            
            chunk_input = torch.tensor(data[i : i + chunk_size])
            chunk_target = torch.tensor(data[i + 1 : i + chunk_size + 1])
            
            logits, _ = model(chunk_input, node_ids, edges, h)
            loss = F.cross_entropy(logits, chunk_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            chunks_count += 1
            
            # Print progress every 100 chunks
            if chunks_count % 100 == 0:
                print(f"Epoch {epoch+1} | Progress: {i/len(data)*100:.1f}% | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / chunks_count if chunks_count > 0 else 0
        print(f"--- Epoch {epoch+1}/{epochs} Complete | Avg Loss: {avg_loss:.4f} ---")
    
    # 5. Final Save
    torch.save(model.state_dict(), weights_path)
    with open(vocab_path, "w", encoding='utf-8') as f:
        json.dump({"w2i": merged_w2i, "i2w": {str(k): v for k, v in merged_i2w.items()}}, f)
    
    print("\n--- KNOWLEDGE UPDATED: Relationships refined without duplicating words ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The text file to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    incremental_train(args.file, args.epochs)
