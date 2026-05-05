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

    def chat(self, prompt, max_len=100, temp=0.8, top_k=5, penalty=1.5):
        # Word-level tokenization
        words = re.findall(r"\w+|[^\w\s]", normalize_text(prompt))
        input_ids = [self.w2i[w] for w in words if w in self.w2i]
        
        if not input_ids: return "Hmm... I don't know those words."

        generated = []
        with torch.no_grad():
            for _ in range(max_len):
                context = torch.tensor(input_ids + generated).unsqueeze(0)
                if context.size(1) > MODEL_CONTEXT: context = context[:, -MODEL_CONTEXT:]
                
                logits, _ = self.model(context)
                next_token_logits = logits[0, -1, :] / temp
                
                # Apply repetition penalty
                for token_id in set(generated):
                    next_token_logits[token_id] /= penalty
                
                # Top-K Sampling for variety
                values, indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(values, dim=-1)
                next_id = indices[torch.multinomial(probs, 1)].item()
                
                word = self.i2w.get(next_id, "")
                if word == "<eos>": break
                generated.append(next_id)
                
                # Stop if repeating too much
                if len(generated) > 10 and len(set(generated[-10:])) < 3: break
        
        return " ".join([self.i2w.get(idx, "") for idx in generated])

async def test_chat():
    brain = PersistentGNNBrain()
    print("\n--- RWKV-7 Apex Long Context Test ---")
    
    prompts = [
        "Tell me about Artificial Intelligence and its applications",
        "Where is New Delhi located and what is famous there?",
        "Explain AI and Delhi in one long sentence"
    ]
    
    for p in prompts:
        print(f"You: {p}")
        ans = brain.chat(p)
        print(f"Brain: {ans}\n")

if __name__ == "__main__":
    asyncio.run(test_chat())
