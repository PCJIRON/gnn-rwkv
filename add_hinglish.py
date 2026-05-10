import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8')

# Ensure dependencies
try:
    import datasets
except ImportError:
    import subprocess
    print("Installing datasets library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"])
    import datasets

from token_graph import TokenGraph

print("="*70)
print("  Adding Hinglish-Everyday-Conversations-1M to Graph")
print("="*70)

# Load existing graph or create new one
GRAPH_PATH = 'graphify-out/v3_graph.json'
try:
    print("Loading existing graph...")
    graph = TokenGraph.load(GRAPH_PATH)
    print(f"Loaded graph with {graph.vocab_size} tokens.")
except Exception as e:
    print(f"Creating new graph (could not load: {e})...")
    graph = TokenGraph(dim=512, window_size=5)

# Load dataset (streaming to avoid downloading everything into RAM at once)
print("\nConnecting to HuggingFace...")
dataset = datasets.load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M", split="train", streaming=True)

# Process a subset first to keep it fast for this session.
# We'll take 10,000 conversations.
num_conversations_to_add = 10000
print(f"\nExtracting {num_conversations_to_add} conversations...")

texts = []
count = 0
t0 = time.time()

for row in dataset:
    # Let's see the structure on the first row
    if count == 0:
        print(f"Data structure: {row.keys()}")
    
    # HuggingFace datasets usually have text in 'text', 'conversations', or similar keys.
    # We'll just dump all string values.
    text_content = ""
    if 'text' in row:
        text_content = str(row['text'])
    elif 'conversation' in row:
        text_content = str(row['conversation'])
    elif 'english' in row and 'hinglish' in row:
        text_content = str(row['hinglish']) + " " + str(row['english'])
    else:
        # Just grab the first string value we find
        for v in row.values():
            if isinstance(v, str):
                text_content += v + " "
    
    if text_content.strip():
        texts.append(text_content.strip())
        count += 1
        
    if count >= num_conversations_to_add:
        break

t1 = time.time()
print(f"Extracted {len(texts)} conversations in {t1-t0:.2f} seconds.")

# Add to graph
print(f"\nAdding {len(texts)} texts to the graph...")
t2 = time.time()
graph.add_texts(texts)
print(f"Added to graph logic. Total raw tokens seen so far: {graph.total_tokens}")

print("\nBuilding Graph Spectral Matrices (dim=512)...")
graph.build(min_freq=3)
t3 = time.time()
print(f"Graph matrices built in {t3-t2:.2f} seconds.")

stats = graph.stats()
print(f"\nNEW GRAPH STATS:")
print(f"  Vocab: {stats['vocab_size']}")
print(f"  Bigrams: {stats['unique_bigrams']}")
print(f"  Total Tokens: {stats['total_tokens']}")

# Save
os.makedirs('graphify-out', exist_ok=True)
graph.save(GRAPH_PATH)
size = os.path.getsize(GRAPH_PATH)
print(f"\nSaved updated graph to {GRAPH_PATH} ({size / 1024 / 1024:.2f} MB).")
print("Done!")
