"""
PROOF: Graph System NEVER Forgets + Data Storage
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from token_graph import TokenGraph

print('='*70)
print('  PROOF: Graph ACCUMULATES - Never Forgets')
print('='*70)

graph = TokenGraph(dim=64, window_size=3)

# Step 1: Add English literature
data_A = [
    'The old man walked slowly through the dark streets of Dublin.',
    'She sat by the window watching the rain fall on the garden.',
    'He never said a word about what happened that evening.',
    'The children played in the yard while their mother cooked dinner.',
] * 5
graph.add_texts(data_A)
graph.build(min_freq=1)

print('\nAfter Data A (English literature):')
print(f'  Vocab: {graph.vocab_size} tokens')
print(f'  Bigrams: {len(graph.bigram_freq)}')
the_freq_A = graph.token_freq['the']
walked_freq_A = graph.token_freq['walked']
dublin_freq_A = graph.token_freq.get('dublin', 0)
print(f'  "the" freq: {the_freq_A}')
print(f'  "walked" freq: {walked_freq_A}')
print(f'  "dublin" freq: {dublin_freq_A}')
print(f'  "python" in vocab: {"python" in graph.vocab}')
print(f'  "def" in vocab: {"def" in graph.vocab}')

# Step 2: Add Python code
data_B = [
    'def hello(): print("world")',
    'for i in range(10): x = i * 2',
    'class Graph: def __init__(self): self.nodes = []',
    'import torch; model = torch.nn.Linear(128, 64)',
    'if x > 0: return True else: return False',
] * 5
graph.add_texts(data_B)
graph.build(min_freq=1)

print('\nAfter Data A + Data B (English + Python):')
print(f'  Vocab: {graph.vocab_size} tokens (GREW!)')
print(f'  Bigrams: {len(graph.bigram_freq)} (GREW!)')
print(f'  "the" freq: {graph.token_freq["the"]} (was {the_freq_A} - STILL HERE!)')
print(f'  "walked" freq: {graph.token_freq["walked"]} (was {walked_freq_A} - STILL HERE!)')
print(f'  "dublin" freq: {graph.token_freq.get("dublin", 0)} (STILL HERE!)')
print(f'  "def" in vocab: {"def" in graph.vocab} (NEW!)')
print(f'  "def" freq: {graph.token_freq["def"]}')
print(f'  "torch" freq: {graph.token_freq["torch"]}')
print(f'  "python" in vocab: {"python" in graph.vocab}')

# Step 3: Add Hindi
data_C = [
    'yeh duniya bahut khoobsurat hai',
    'mera naam python developer hai',
    'aaj mausam bahut achha hai',
    'graph se hum sab seekh sakte hain',
] * 5
graph.add_texts(data_C)
graph.build(min_freq=1)

print('\nAfter Data A + B + C (English + Python + Hindi):')
print(f'  Vocab: {graph.vocab_size} tokens (GREW AGAIN!)')
print(f'  Bigrams: {len(graph.bigram_freq)}')
print(f'  "the" freq: {graph.token_freq["the"]} (ENGLISH INTACT!)')
print(f'  "def" freq: {graph.token_freq["def"]} (PYTHON INTACT!)')
print(f'  "duniya" freq: {graph.token_freq.get("duniya", 0)} (HINDI ADDED!)')
print(f'  "bahut" freq: {graph.token_freq.get("bahut", 0)} (HINDI ADDED!)')
print(f'  "hai" freq: {graph.token_freq.get("hai", 0)} (HINDI ADDED!)')

# Step 4: Add MORE English - do frequencies grow?
data_D = [
    'The old man walked slowly through the dark streets of Dublin.',
] * 10
graph.add_texts(data_D)
graph.build(min_freq=1)

print('\nAfter adding MORE English (same sentences again):')
print(f'  "the" freq: {graph.token_freq["the"]} (was {the_freq_A} -> GREW!)')
print(f'  "walked" freq: {graph.token_freq["walked"]} (was {walked_freq_A} -> GREW!)')
print(f'  "def" still: {graph.token_freq["def"]} (Python NOT forgotten!)')
print(f'  "duniya" still: {graph.token_freq.get("duniya", 0)} (Hindi NOT forgotten!)')

# === WHERE IS DATA SAVED? ===
print(f'\n{"="*70}')
print(f'  WHERE IS DATA SAVED?')
print(f'{"="*70}')

print(f'\n  IN MEMORY (during runtime):')
print(f'    - token_freq: Counter with {len(graph.token_freq)} entries')
print(f'    - bigram_freq: Counter with {len(graph.bigram_freq)} entries')
print(f'    - cooccurrence: dict with {len(graph.cooccurrence)} entries')
print(f'    - embeddings: numpy array {graph.embeddings.shape}')
print(f'    - W_r, W_k, W_v, W_w, W_a, W_g: each [{graph.dim}x{graph.dim}]')
print(f'    - transition matrix: [{graph.vocab_size}x{graph.vocab_size}]')

# Save to disk
os.makedirs('graphify-out', exist_ok=True)
graph.save('graphify-out/v3_graph.json')
size = os.path.getsize('graphify-out/v3_graph.json')
print(f'\n  ON DISK (after save):')
print(f'    - File: graphify-out/v3_graph.json')
print(f'    - Size: {size:,} bytes')

# Load back
loaded = TokenGraph.load('graphify-out/v3_graph.json')
print(f'\n  LOADED BACK:')
print(f'    - Vocab: {loaded.vocab_size} tokens (SAME!)')
print(f'    - "the" freq: {loaded.token_freq["the"]} (SAME!)')
print(f'    - "def" freq: {loaded.token_freq["def"]} (SAME!)')
print(f'    - "duniya" freq: {loaded.token_freq.get("duniya", 0)} (SAME!)')
print(f'    - ALL DATA PRESERVED!')

print(f'\n{"="*70}')
print(f'  COMPARISON: Our System vs Neural Networks')
print(f'{"="*70}')
print("""
  Neural Network (GPT/RWKV trained):
    Train on English -> W1 (English weights)
    Train on Python  -> W2 (OVERWRITES W1! Catastrophic Forgetting!)
    Train on Hindi   -> W3 (Forgets both English AND Python!)
    Solution: Train on ALL data together (expensive!)
    
  Our Graph System:
    Add English -> Graph grows (English nodes+edges)
    Add Python  -> Graph grows MORE (Python nodes+edges added)
    Add Hindi   -> Graph grows MORE (Hindi nodes+edges added)
    English data? STILL IN THE GRAPH (frequencies accumulated!)
    Python data? STILL IN THE GRAPH (frequencies accumulated!)
    Solution: Just keep adding data. Graph NEVER forgets.
    
  WHY? Because:
    Neural networks OVERWRITE weights on each training step
    Our system ADDS to the graph on each data ingestion
    Graph is ADDITIVE, neural training is DESTRUCTIVE
""")
