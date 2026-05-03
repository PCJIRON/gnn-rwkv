import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNRWKVCell(nn.Module):
    """
    A custom GNN cell inspired by RWKV (Receptance Weighted Key Value).
    It replaces standard Transformer attention with Graph-based Recurrence.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # RWKV-style parameters: Receptance, Key, Value
        self.w_receptance = nn.Linear(dim, dim)
        self.w_key = nn.Linear(dim, dim)
        self.w_value = nn.Linear(dim, dim)
        
        # Graph Mixing parameters
        self.gate = nn.Linear(dim, dim)
        self.time_decay = nn.Parameter(torch.ones(dim)) # Similar to RWKV 'w'
        self.time_first = nn.Parameter(torch.ones(dim)) # Similar to RWKV 'u'

    def forward(self, x, edge_index, hidden_state):
        # x: [num_nodes, dim] - Current features of characters
        # edge_index: [2, num_edges] - Social connections
        # hidden_state: [num_nodes, dim] - Past memory of the graph
        
        r = torch.sigmoid(self.w_receptance(x))
        k = self.w_key(x)
        v = self.w_value(x)
        
        # Graph-Mixing: Aggregate 'Key' and 'Value' from neighbors
        # (Instead of attending to everyone, we only mix with friends in the graph)
        row, col = edge_index
        
        # Simple message passing: Each node 'hears' its neighbors
        neighbor_k = torch.zeros_like(k)
        neighbor_v = torch.zeros_like(v)
        neighbor_k.index_add_(0, row, k[col])
        neighbor_v.index_add_(0, row, v[col])
        
        # RWKV-style WKV calculation over the graph
        # State update: h_new = decay * h_old + K_neighbor * V_neighbor
        new_h = (self.time_decay.exp() * hidden_state) + (neighbor_k * neighbor_v)
        
        # Output: r * h
        out = r * new_h
        return out, new_h

class SocialRWKVBrain(nn.Module):
    def __init__(self, num_entities, dim=64):
        super().__init__()
        self.dim = dim
        self.embeddings = nn.Embedding(num_entities, dim)
        self.cell = GNNRWKVCell(dim)
        self.output_head = nn.Linear(dim, num_entities) # Predict next active character

    def forward(self, entity_ids, edge_index, hidden_states):
        x = self.embeddings(entity_ids)
        out, new_h = self.cell(x, edge_index, hidden_states)
        logits = self.output_head(out)
        return logits, new_h

# --- TESTING THE CUSTOM GNN-RWKV BRAIN ---
def test_gnn_rwkv():
    print("--- Initializing Custom GNN-RWKV Architecture ---")
    num_chars = 5 # Vance, Arthur, Silas, Genevieve, Maya
    dim = 16
    
    model = SocialRWKVBrain(num_chars, dim)
    
    # Initial State (Zero memory)
    h = torch.zeros((num_chars, dim))
    
    # Input: Character IDs
    ids = torch.tensor([0, 1, 2, 3, 4])
    
    # Graph: (0 is connected to 1, 1 is connected to 2, etc.)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], 
                          [1, 0, 2, 1, 4, 3]], dtype=torch.long)
    
    print("Step 1: Processing Social Interactions...")
    logits, h = model(ids, edges, h)
    
    print(f"Graph State Updated. Hidden state shape: {h.shape}")
    print("Logits (Next likely interaction):")
    print(F.softmax(logits, dim=-1))
    
    print("\n--- Success: Custom GNN-RWKV Brain is Operational ---")
    print("This model works with O(E) complexity and maintains a recurrent social memory.")

if __name__ == "__main__":
    test_gnn_rwkv()
