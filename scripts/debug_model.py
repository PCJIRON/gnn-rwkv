import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from src.gnn_rwkv_story_gen import SocialNarrativeModel, MODEL_DIM, MODEL_NODES, MODEL_LAYERS, MODEL_CONTEXT
import json

with open('vocab.json', 'r') as f:
    vocab = json.load(f)

w2i = vocab.get('w2i', {})
i2w = {int(k): v for k, v in vocab.get('i2w', {}).items()}
vocab_size = len(w2i)

model = SocialNarrativeModel(vocab_size, MODEL_NODES, MODEL_DIM, MODEL_LAYERS, MODEL_CONTEXT)
model.load_state_dict(torch.load('social_brain.pth', map_location='cpu'))
model.eval()

test_input = 'hello'
input_ids = [w2i.get(w, w2i.get('<unk>', 1)) for w in test_input.lower().split()]
input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

node_ids = torch.arange(MODEL_NODES, dtype=torch.long)
h = [torch.zeros((MODEL_NODES, MODEL_DIM)) for _ in range(MODEL_LAYERS)]

with torch.no_grad():
    logits, h = model(input_tensor, node_ids, None, h)
    probs = F.softmax(logits[0, -1], dim=-1)
    top_prob, top_idx = probs.max(dim=-1)

print(f'Input: {test_input}')
print(f'Top token: {i2w.get(top_idx.item(), "?")} (prob: {top_prob.item():.4f})')
print('Top 5 tokens:')
top5_vals, top5_idx = probs.topk(5)
for idx, prob in zip(top5_idx, top5_vals):
    print(f'  {i2w.get(idx.item(), "?"):15} : {prob.item():.4f}')