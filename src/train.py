import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import argparse
import shutil
import time
import random
from pathlib import Path
from gnn_rwkv_story_gen import (
    MODEL_CONTEXT,
    MODEL_DIM,
    MODEL_LAYERS,
    MODEL_NODES,
    SocialNarrativeModel,
    TOKENIZER_VERSION,
    get_data,
    get_data_from_text,
)

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

WEIGHTS_PATH = "social_brain.pth"
VOCAB_PATH = "vocab.json"
META_PATH = "brain_meta.json"
ACTIVE_BRAIN_PATH = "active_brain.json"
BACKUP_DIR = Path("brain_backups")
BASE_CORPUS_PATH = "base_language_corpus.txt"
WEIGHTS_DIR = Path("weights")

def _backup_existing_brain():
    if not (os.path.exists(WEIGHTS_PATH) or os.path.exists(VOCAB_PATH) or os.path.exists(META_PATH)):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = BACKUP_DIR / stamp
    target.mkdir(parents=True, exist_ok=True)
    for path in [WEIGHTS_PATH, VOCAB_PATH, META_PATH, ACTIVE_BRAIN_PATH]:
        if os.path.exists(path):
            shutil.copy2(path, target / Path(path).name)
    return str(target)

def _write_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def _save_weights_versioned(model):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_DIR / f"social_brain_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), target)
    return str(target)

def _save_active_manifest(metadata, backup_path=None, weights_path=WEIGHTS_PATH):
    manifest = {
        "weights": weights_path,
        "vocab": VOCAB_PATH,
        "meta": META_PATH,
        "tokenizer": TOKENIZER_VERSION,
        "dim": metadata["dim"],
        "layers": metadata["layers"],
        "num_nodes": metadata["num_nodes"],
        "context": metadata["context"],
        "vocab_size": metadata["vocab_size"],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": metadata.get("source_file"),
        "backup": backup_path,
    }
    _write_json_atomic(ACTIVE_BRAIN_PATH, manifest)

def _load_active_paths():
    if not os.path.exists(ACTIVE_BRAIN_PATH):
        return WEIGHTS_PATH, VOCAB_PATH, META_PATH
    try:
        with open(ACTIVE_BRAIN_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return (
            manifest.get("weights", WEIGHTS_PATH),
            manifest.get("vocab", VOCAB_PATH),
            manifest.get("meta", META_PATH),
        )
    except (OSError, json.JSONDecodeError):
        return WEIGHTS_PATH, VOCAB_PATH, META_PATH

def _read_limited_training_text(input_file, max_sentences=None):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if max_sentences is not None:
        lines = lines[:max_sentences]
    return "\n".join(lines).lower()

def train_brain(
    input_file,
    epochs=50,
    persist=True,
    max_sentences=None,
    progress_callback=None,
    verbose=True,
    reset_existing=False,
    batch_size=8
):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return None

    active_weights, active_vocab, active_meta = _load_active_paths()
    weights_path = active_weights if persist and not reset_existing else WEIGHTS_PATH
    vocab_path = active_vocab if persist and not reset_existing else VOCAB_PATH
    meta_path = active_meta if persist and not reset_existing else META_PATH
    
    # 1. Vocab & Data Preparation
    training_text = _read_limited_training_text(input_file, max_sentences=max_sentences)
    new_data_raw, new_w2i, new_i2w = get_data_from_text(training_text)
    vocab_size = len(new_w2i)
    
    # 2. Model Init
    model = SocialNarrativeModel(vocab_size, MODEL_NODES, dim=MODEL_DIM, num_layers=MODEL_LAYERS, max_seq=MODEL_CONTEXT)
    
    if persist and not reset_existing and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict, strict=False)
        except:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Pre-calculate Batched Node IDs
    node_ids = torch.arange(MODEL_NODES).unsqueeze(0).expand(batch_size, -1) # [B, Nodes]
    
    chunk_size = 64
    input_ids = torch.tensor(new_data_raw)
    
    if verbose:
        print(f"Starting Vectorized Training: {epochs} epochs, Batch Size: {batch_size}")
    
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        h = [torch.zeros((batch_size, MODEL_NODES, MODEL_DIM)) for _ in range(model.num_layers)]
        
        # Simple Batched Loader
        # We divide the input ids into batch_size chunks
        num_chunks = len(input_ids) // (batch_size * chunk_size)
        for i in range(num_chunks):
            # Create a batch of chunks
            batch_x = []
            batch_y = []
            for b in range(batch_size):
                start_idx = i * chunk_size + (b * (len(input_ids) // batch_size))
                batch_x.append(input_ids[start_idx : start_idx + chunk_size])
                batch_y.append(input_ids[start_idx + 1 : start_idx + chunk_size + 1])
            
            x_tensor = torch.stack(batch_x) # [B, Seq]
            y_tensor = torch.stack(batch_y) # [B, Seq]
            
            optimizer.zero_grad()
            logits, h, _ = model.forward(x_tensor, node_ids, None, h)
            
            # Detach h for next step (TBPTT)
            h = [s.detach() for s in h]
            
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_tensor.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_chunks)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            
    if persist:
        torch.save(model.state_dict(), weights_path)
        vocab_data = {"tokenizer": TOKENIZER_VERSION, "w2i": new_w2i, "i2w": {str(k): v for k, v in new_i2w.items()}}
        _write_json_atomic(vocab_path, vocab_data)
        metadata = {"dim": MODEL_DIM, "layers": MODEL_LAYERS, "num_nodes": MODEL_NODES, "context": MODEL_CONTEXT, "vocab_size": vocab_size, "tokenizer": TOKENIZER_VERSION}
        _write_json_atomic(meta_path, metadata)
        _save_active_manifest(metadata)
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-sentences", type=int, default=500)
    args = parser.parse_args()
    
    train_brain(args.file, epochs=args.epochs, max_sentences=args.max_sentences)
