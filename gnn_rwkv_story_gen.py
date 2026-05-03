import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import collections

# UTF-8 for terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class GNNRWKVCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w_receptance = nn.Linear(dim, dim)
        self.w_key = nn.Linear(dim, dim)
        self.w_value = nn.Linear(dim, dim)
        self.time_decay = nn.Parameter(torch.ones(dim) * -0.5) # Initial decay
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x, edge_index, h):
        # RWKV-style Receptance, Key, Value
        r = torch.sigmoid(self.w_receptance(x))
        k = self.w_key(x)
        v = self.w_value(x)
        
        # Graph-Mixing Logic
        row, col = edge_index
        neighbor_kv = torch.zeros_like(k)
        if edge_index.shape[1] > 0:
            # Weighted interaction based on graph structure
            neighbor_kv.index_add_(0, row, k[col] * v[col])
            
        # State Update: h = decay * h + interaction
        new_h = (torch.exp(self.time_decay) * h) + neighbor_kv
        
        # Output gating
        out = r * self.ln(new_h)
        return out, new_h

class SocialNarrativeModel(nn.Module):
    def __init__(self, vocab_size, num_nodes, dim=64):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, dim)
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.cell = GNNRWKVCell(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, word_ids, node_ids, edge_index, h):
        w_x = self.word_emb(word_ids) # Word embeddings
        n_x = self.node_emb(node_ids) # Graph node embeddings
        
        # Process graph recurrent state
        graph_out, new_h = self.cell(n_x, edge_index, h)
        
        # Fuse word with average global graph state
        # (This acts as the 'World State' for the storyteller)
        world_state = graph_out.mean(dim=0).unsqueeze(0)
        combined = w_x + world_state
        
        logits = self.head(combined)
        return logits, new_h

def get_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Clean and tokenize
    text = text.replace(".", " . ").replace(",", " , ").replace('"', ' " ')
    tokens = text.split()
    vocab = sorted(list(set(tokens)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    data = [w2i[t] for t in tokens]
    return data, w2i, i2w

def train_and_storytelling():
    data, w2i, i2w = get_data("story.txt")
    vocab_size = len(w2i)
    num_nodes = 5 # Arthur, Vance, Silas, Genevieve, Maya
    dim = 64
    
    model = SocialNarrativeModel(vocab_size, num_nodes, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Complex Social Graph for training
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], 
                          [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    node_ids = torch.tensor([0, 1, 2, 3, 4])
    
    print(f"--- Training Deep GNN-RWKV Brain ({vocab_size} vocab) ---")
    
    model.train()
    for epoch in range(500):
        h = torch.zeros((num_nodes, dim))
        optimizer.zero_grad()
        
        inputs = torch.tensor(data[:-1])
        targets = torch.tensor(data[1:])
        
        logits, _ = model(inputs, node_ids, edges, h)
        loss = F.cross_entropy(logits, targets)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Narrative Loss: {loss.item():.4f}")

    # STORY GENERATION
    print("\n" + "="*50)
    print("GENERATED EPIC STORY (GNN-RWKV):")
    print("="*50)
    
    model.eval()
    h = torch.zeros((num_nodes, dim))
    current_word_id = data[0] # "in"
    generated_ids = [current_word_id]
    
    # Avoid repetition penalty
    counts = collections.defaultdict(int)
    
    with torch.no_grad():
        for i in range(10): # Let's look at 10 steps in detail
            word_tensor = torch.tensor([current_word_id])
            logits, h = model(word_tensor, node_ids, edges, h)
            
            # Show the "Brain's" internal probabilities
            probs = F.softmax(logits[0], dim=-1)
            top_probs, top_indices = torch.topk(probs, 3)
            
            print(f"Step {i+1} | Current: '{i2w[current_word_id]}' | Thinking of next word...")
            for p, idx in zip(top_probs, top_indices):
                print(f"  -> '{i2w[idx.item()]}' (Probability: {p.item()*100:.2f}%)")
            
            # Sample next word (with small penalty for variety)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)
            current_word_id = next_id
            print("-" * 30)
            
    story = " ".join([i2w[i] for i in generated_ids])
    # Basic formatting
    story = story.replace(" . ", ". ").replace(" , ", ", ")
    print(story)
    print("="*50)

if __name__ == "__main__":
    train_and_storytelling()
