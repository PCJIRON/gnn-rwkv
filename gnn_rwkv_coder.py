import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class GNNRWKVCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w_receptance = nn.Linear(dim, dim)
        self.w_key = nn.Linear(dim, dim)
        self.w_value = nn.Linear(dim, dim)
        self.time_decay = nn.Parameter(torch.ones(dim) * -0.5)
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x, edge_index, h):
        r = torch.sigmoid(self.w_receptance(x))
        k = self.w_key(x)
        v = self.w_value(x)
        
        row, col = edge_index
        neighbor_kv = torch.zeros_like(k)
        if edge_index.shape[1] > 0:
            neighbor_kv.index_add_(0, row, k[col] * v[col])
            
        new_h = (torch.exp(self.time_decay) * h) + neighbor_kv
        out = r * self.ln(new_h)
        return out, new_h

class MicroCoderModel(nn.Module):
    def __init__(self, vocab_size, num_nodes, dim=64):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, dim)
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.cell = GNNRWKVCell(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, word_ids, node_ids, edge_index, h):
        w_x = self.word_emb(word_ids)
        n_x = self.node_emb(node_ids)
        graph_out, new_h = self.cell(n_x, edge_index, h)
        
        # Merge word state with global code-graph state
        context = graph_out.mean(dim=0).unsqueeze(0)
        logits = self.head(w_x + context)
        return logits, new_h

def tokenize_code(text):
    # Precise tokenization for symbols and keywords
    tokens = re.findall(r'[a-zA-Z_]\w*|[\d]+|[^\w\s]', text)
    return tokens

def train_and_code():
    with open("code_corpus.txt", "r") as f:
        code_text = f.read()
    
    tokens = tokenize_code(code_text)
    vocab = sorted(list(set(tokens)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    data = [w2i[t] for t in tokens]
    
    vocab_size = len(vocab)
    num_nodes = 5 # Keywords: def, print, return, for, if
    dim = 64
    
    model = MicroCoderModel(vocab_size, num_nodes, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # GNN: def(0) connects to symbols(1), print(2) connects to variables(3)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]], dtype=torch.long)
    node_ids = torch.tensor([0, 1, 2, 3, 4])
    
    print(f"--- Training Micro-Coder GNN-RWKV ({vocab_size} symbols) ---")
    
    model.train()
    for epoch in range(400):
        h = torch.zeros((num_nodes, dim))
        optimizer.zero_grad()
        
        inputs = torch.tensor(data[:-1])
        targets = torch.tensor(data[1:])
        
        logits, _ = model(inputs, node_ids, edges, h)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Coding Loss: {loss.item():.4f}")

    # TEST: GENERATE NEW CODE
    print("\n--- Generating 'New' Python Code ---")
    model.eval()
    h = torch.zeros((num_nodes, dim))
    
    # Let's prompt it with "def" and see what it does
    current_id = w2i["def"]
    generated = [i2w[current_id]]
    
    with torch.no_grad():
        for _ in range(30):
            word_tensor = torch.tensor([current_id])
            logits, h = model(word_tensor, node_ids, edges, h)
            
            # Repetition penalty for cleaner code
            for idx in generated[-3:]: 
                logits[0][w2i[idx]] -= 1.0
                
            probs = F.softmax(logits[0] / 0.5, dim=-1) # Lower temp for strict syntax
            next_id = torch.multinomial(probs, 1).item()
            
            generated.append(i2w[next_id])
            current_id = next_id
            
    print(" ".join(generated))

if __name__ == "__main__":
    train_and_code()
