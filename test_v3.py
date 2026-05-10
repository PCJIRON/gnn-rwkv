"""
V3 Test: dim=512 + 4-Head WKV (RWKV-native, not transformer)
"""
import sys, os, glob, time
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn.functional as F
import numpy as np

from token_graph import TokenGraph
from graph_rwkv7_v3 import MultiHeadGraphWKV, InfiniteContextGeneratorV3

print('='*70)
print('  V3: dim=512, 4-Head WKV, RWKV-native Multi-Head')
print('='*70)

# Load Dubliners
files = glob.glob('James*')
with open(files[0], 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(texts)} segments\n")

# Build graph with dim=512
print("Building token graph (dim=512)...")
t0 = time.time()
graph = TokenGraph(dim=512, window_size=5)
graph.add_texts(texts)
graph.build(min_freq=3)
t1 = time.time()

stats = graph.stats()
print(f"  Built in {t1-t0:.2f}s")
print(f"  Vocab: {stats['vocab_size']}")
print(f"  Bigrams: {stats['unique_bigrams']}")
print(f"  Communities: {stats['communities']}")
print(f"  Embedding: {stats['embedding_shape']}")

# Build V3 engine
print(f"\nBuilding 4-head WKV engine...")
t2 = time.time()
gen = InfiniteContextGeneratorV3(graph, n_heads=4)
t3 = time.time()
print(f"  Built in {t3-t2:.2f}s")
print(f"  Head dim: {gen.rwkv.head_dim}")
print(f"  Total dim: {gen.rwkv.dim}")
print(f"  Heads: {gen.rwkv.n_heads}")
print(f"  State per head: [{gen.rwkv.head_dim}x{gen.rwkv.head_dim}]")
print(f"  Weight matrices per head: 6 x [{gen.rwkv.head_dim}x{gen.rwkv.head_dim}]")
total_params = gen.rwkv.n_heads * 6 * gen.rwkv.head_dim ** 2 + gen.rwkv.dim ** 2
print(f"  Total parameters: {total_params:,}")

# Context sensitivity
print(f"\n{'='*70}")
print(f"  Context Sensitivity Test")
print(f"{'='*70}")

rwkv = MultiHeadGraphWKV(graph, n_heads=4)
for w in ['the', 'old', 'man']:
    if w in graph.vocab: rwkv.step(graph.vocab[w])
out_A = rwkv.step(graph.vocab.get('said', 0))

rwkv2 = MultiHeadGraphWKV(graph, n_heads=4)
for w in ['she', 'never']:
    if w in graph.vocab: rwkv2.step(graph.vocab[w])
out_B = rwkv2.step(graph.vocab.get('said', 0))

cos = F.cosine_similarity(out_A.unsqueeze(0), out_B.unsqueeze(0)).item()
print(f"  Cosine sim (same word, different context): {cos:.4f}")

logits_A = rwkv.compute_logits(out_A)
logits_B = rwkv2.compute_logits(out_B)
topA = torch.topk(logits_A, 5)
topB = torch.topk(logits_B, 5)

print(f'\n  "the old man said" -> next:')
for v, i in zip(topA.values, topA.indices):
    t = graph.id_to_token.get(i.item(), '?')
    bg = graph.bigram_freq.get(('said', t), 0)
    print(f'    {t:15s} score={v.item():.4f} (bigram: {bg}x)')

print(f'\n  "she never said" -> next:')
for v, i in zip(topB.values, topB.indices):
    t = graph.id_to_token.get(i.item(), '?')
    bg = graph.bigram_freq.get(('said', t), 0)
    print(f'    {t:15s} score={v.item():.4f} (bigram: {bg}x)')

# Generation
print(f"\n{'='*70}")
print(f"  Text Generation (4-Head WKV, dim=512)")
print(f"{'='*70}")

gen = InfiniteContextGeneratorV3(graph, n_heads=4)

prompts = [
    "the old man",
    "she was afraid",
    "he came home",
    "dublin was",
    "the night was cold",
    "they walked through",
    "i want to go",
    "gabriel looked at",
]

for p in prompts:
    r = gen.generate(prompt=p, max_tokens=60, temperature=0.65, 
                     top_k=15, top_p=0.9, seed=42)
    print(f"\n  [{p}]")
    print(f"  -> {r}")

# Infinite context session
print(f"\n{'='*70}")
print(f"  Infinite Context Session")
print(f"{'='*70}")

session = InfiniteContextGeneratorV3(graph, n_heads=4)

turns = [
    "the old man walked into the room",
    "she looked at him and said",
    "he never came back",
    "the door was closed",
]

for i, turn in enumerate(turns):
    r = session.generate(prompt=turn, max_tokens=30, temperature=0.65, seed=42)
    s = session.session_stats()
    print(f"\n  Turn {i+1}: [{turn}]")
    print(f"  -> {r}")
    print(f"  State: steps={s['step_count']}, tokens={s['session_tokens']}, "
          f"norms={[f'{n:.2f}' for n in s['state_norms']]}")

print(f"\n{'='*70}")
print("  Done!")
print(f"{'='*70}")
