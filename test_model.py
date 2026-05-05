import sys, json, torch
sys.stdout.reconfigure(encoding='utf-8')

from src.gnn_rwkv_story_gen import SocialNarrativeModel, get_data_from_text, MODEL_DIM, MODEL_NODES, MODEL_LAYERS, MODEL_CONTEXT

with open('vocab.json') as f:
    vd = json.load(f)
    w2i = vd['w2i']
i2w = {int(k): v for k, v in vd.get('i2w', {}).items()}
vocab_size = len(w2i)

model = SocialNarrativeModel(vocab_size, MODEL_NODES, MODEL_DIM, MODEL_LAYERS, MODEL_CONTEXT)
model.load_state_dict(torch.load('social_brain.pth', map_location='cpu'))
model.eval()

print('=== Model Predictions ===')

for prompt in ['python', 'learn', 'code', 'programming', 'ai']:
    data, _, _ = get_data_from_text(prompt)
    while len(data) < 16:
        data.append(0)
    data = data[:16]
    
    word_ids = torch.tensor(data, dtype=torch.long)
    node_ids = torch.arange(MODEL_NODES, dtype=torch.long)
    h = [torch.zeros((MODEL_NODES, MODEL_DIM)) for _ in range(MODEL_LAYERS)]
    
    with torch.no_grad():
        logits, h = model(word_ids, node_ids, None, h)
    
    top_idx = torch.argsort(logits[-1], descending=True)[:10]
    top_words = [i2w.get(t.item(), str(t.item())) for t in top_idx]
    
    print(f'Prompt: "{prompt}" -> {top_words}')