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

    def chat(self, prompt, max_len=40):
        # Word-level tokenization
        words = re.findall(r"\w+|[^\w\s]", normalize_text(prompt))
        input_ids = [self.w2i[w] for w in words if w in self.w2i]
        
        if not input_ids: return "..."

        generated = []
        with torch.no_grad():
            for _ in range(max_len):
                context = torch.tensor(input_ids + generated).unsqueeze(0)
                if context.size(1) > MODEL_CONTEXT: context = context[:, -MODEL_CONTEXT:]
                
                logits, _ = self.model(context)
                next_id = torch.argmax(logits[0, -1, :]).item()
                
                word = self.i2w.get(next_id, "<unk>")
                if word == "<eos>": break
                generated.append(next_id)
                if len(generated) > 1 and word == generated[-1]: break
        
        return " ".join([self.i2w.get(idx, "") for idx in generated])

async def test_chat():
    brain = PersistentGNNBrain()
    print("\n--- Generative Brain v4.0 Chat Test ---")
    questions = ["What is AI?", "New Delhi is capital", "Evolution is"]
    for q in questions:
        print(f"You: {q}")
        ans = brain.chat(q)
        print(f"Brain: {ans}\n")

if __name__ == "__main__":
    asyncio.run(test_chat())
