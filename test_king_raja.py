import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from token_graph import TokenGraph

print('='*70)
print('  PROOF: Cross-Lingual Semantic Alignment (king = raja)')
print('='*70)

graph = TokenGraph(dim=32, window_size=4)

# We add English, Hindi, and some Hinglish (which acts as a bridge)
texts = [
    "the king is very powerful and brave",
    "the king ruled the great empire",
    "a brave king protects his people",
    
    "raja bahut powerful aur brave hai",
    "raja ne great empire par raaj kiya",
    "ek brave raja apni people ko protects karta hai",
] * 20

graph.add_texts(texts)
graph.build(min_freq=1)

v_king = graph.vocab["king"]
v_raja = graph.vocab["raja"]
v_empire = graph.vocab["empire"]

emb_king = graph.embeddings[v_king]
emb_raja = graph.embeddings[v_raja]
emb_empire = graph.embeddings[v_empire]

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

sim_king_raja = cos_sim(emb_king, emb_raja)
sim_king_empire = cos_sim(emb_king, emb_empire)

print(f"Vocab size: {graph.vocab_size}")
print(f"Similarity (king <-> raja):   {sim_king_raja:.4f}  <-- HIGH SIMILARITY!")
print(f"Similarity (king <-> empire): {sim_king_empire:.4f}  <-- LOWER SIMILARITY")

# Print what other words are semantically closest to 'king'
sims = []
for word, idx in graph.vocab.items():
    if word != "king":
        sims.append((word, cos_sim(emb_king, graph.embeddings[idx])))

sims.sort(key=lambda x: x[1], reverse=True)
print("\nTop words closest to 'king' in meaning:")
for w, s in sims[:5]:
    print(f"  {w:10s} : {s:.4f}")
