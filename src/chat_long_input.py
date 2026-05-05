import torch
import torch.nn.functional as F
import json
import os
import sys
import asyncio
import re
from gnn_rwkv_story_gen import (
    SocialNarrativeModel,
    normalize_text,
    MODEL_DIM,
    MODEL_LAYERS,
    MODEL_NODES,
    MODEL_CONTEXT,
)

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

WEIGHTS_PATH = "social_brain.pth"
VOCAB_PATH = "vocab.json"

class PersistentGNNBrain:
    def __init__(self):
        self.load_brain()

    def load_brain(self):
        if not os.path.exists(VOCAB_PATH): return
        with open(VOCAB_PATH, "r", encoding='utf-8') as f:
            v_data = json.load(f)
            self.w2i = v_data["w2i"]
            self.i2w = {int(k): v for k, v in v_data["i2w"].items()}
        
        self.vocab_size = len(self.w2i)
        self.model = SocialNarrativeModel(self.vocab_size)
        if os.path.exists(WEIGHTS_PATH):
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"), strict=False)
        self.model.eval()

    def chat(self, prompt, max_len=120, temp=0.8, top_k=5, penalty=1.2):
        words = re.findall(r"\w+|[^\w\s]", normalize_text(prompt))
        input_ids = [self.w2i[w] for w in words if w in self.w2i]
        
        if not input_ids: return "..."

        generated = []
        with torch.no_grad():
            for _ in range(max_len):
                context = torch.tensor(input_ids + generated).unsqueeze(0)
                if context.size(1) > MODEL_CONTEXT: context = context[:, -MODEL_CONTEXT:]
                
                logits, _ = self.model(context)
                next_token_logits = logits[0, -1, :] / temp
                
                # Stability Clamp
                next_token_logits = torch.clamp(next_token_logits, -10.0, 10.0)
                
                # Penalty
                for token_id in set(generated):
                    next_token_logits[token_id] /= penalty
                
                values, indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(values, dim=-1)
                
                # Final NaN check
                if torch.isnan(probs).any():
                    next_id = torch.argmax(next_token_logits).item()
                else:
                    next_id = indices[torch.multinomial(probs, 1)].item()
                
                word = self.i2w.get(next_id, "")
                if word == "<eos>": break
                generated.append(next_id)
                if len(generated) > 20 and len(set(generated[-15:])) < 3: break
        
        return " ".join([self.i2w.get(idx, "") for idx in generated])

async def test_long_input():
    brain = PersistentGNNBrain()
    print("\n--- RWKV-7 Apex Long Input Test ---")
    
    long_prompt = "Artificial Intelligence is the future and machines are learning like humans. I live in New Delhi which is the capital of India. Can you tell me more about how AI works in our dimaag and how Delhi is different from other cities?"
    
    print(f"You (Long Prompt): {long_prompt}\n")
    ans = brain.chat(long_prompt)
    print(f"Brain: {ans}\n")

if __name__ == "__main__":
    asyncio.run(test_long_input())
