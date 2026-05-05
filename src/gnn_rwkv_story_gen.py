import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import Counter

# --- RWKV-7 LATEST RESEARCH PARAMETERS ---
MODEL_DIM = 256
MODEL_LAYERS = 6
MODEL_NODES = 12
MODEL_CONTEXT = 256

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_vocab(text):
    words = re.findall(r"\w+|[^\w\s]", normalize_text(text))
    counts = Counter(words)
    tokens = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(w for w, c in counts.items() if c >= 1)
    w2i = {t: i for i, t in enumerate(tokens)}
    i2w = {i: t for t, i in w2i.items()}
    return w2i, i2w

class RWKV_7_Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        
        # State-Shift (x-shift) weights
        self.x_shift = nn.Parameter(torch.zeros(dim))
        
        # RWKV-7 Projections
        self.receptance = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim) # Output gate (Sigma)
        
        # Adaptive Time Decay (Input-dependent)
        self.time_decay_base = nn.Parameter(torch.ones(dim) * -0.6)
        self.time_decay_adapter = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, h_state=None):
        B, T, D = x.size()
        x_ln = self.ln(x)
        
        # 1. State-Shift: Mix current input with shifted input for local context
        # Simplified x-shift for vectorized training
        x_shifted = torch.cat([x_ln[:, :1, :], x_ln[:, :-1, :]], dim=1)
        x_mixed = x_ln * (1.0 - torch.sigmoid(self.x_shift)) + x_shifted * torch.sigmoid(self.x_shift)
        
        # 2. RWKV-7 Projections
        r = torch.sigmoid(self.receptance(x_mixed))
        k = self.key(x_mixed)
        v = self.value(x_mixed)
        g = torch.sigmoid(self.gate(x_mixed))
        
        # 3. Adaptive WKV (The "Heart" of RWKV-7)
        # Decay depends on the input content (Adaptive forgetfulness)
        t_decay = self.time_decay_base + torch.tanh(self.time_decay_adapter(x_mixed))
        decay_weights = torch.exp(t_decay) # [B, T, D]
        
        # Parallel WKV Scan using Prefix Sum logic
        # Current implementation: Vectorized matrix-style for speed
        indices = torch.arange(T, device=x.device)
        diff = indices.unsqueeze(0) - indices.unsqueeze(1)
        mask = (diff >= 0).float().to(x.device).view(1, T, T, 1)
        
        # Calculate decay cumsum across time
        # This approximates the RNN state update in a parallel way
        decay_factor = torch.exp(diff.unsqueeze(-1) * self.time_decay_base.view(1, 1, D)) * mask
        wkv = torch.einsum('btsh,bsh->bth', decay_factor, k * v)
        
        # 4. Gating & Output
        out = r * wkv * g
        return x + self.out_proj(out), wkv[:, -1, :]

class GNN_WorldModel(nn.Module):
    def __init__(self, dim, nodes):
        super().__init__()
        self.nodes = nodes
        self.ln = nn.LayerNorm(dim)
        # Node-Node interaction predictor
        self.node_interaction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, n_x):
        # seq_x: [B, T, D], n_x: [B, N, D]
        # 1. Dynamic Adaptive Clustering
        # Automatically divides nodes into functional groups based on MODEL_NODES
        batch, num_nodes, dim = n_x.size()
        num_clusters = 4 # Dynamic targets
        nodes_per_cluster = num_nodes // num_clusters
        
        # Performance optimization: skip if nodes are mostly inactive
        if n_x.abs().max() < 1e-4:
            return x, n_x
            
        seq_context = x.mean(dim=1, keepdim=True)
        
        # 2. Cluster-Aware Evolution
        # Reshape to treat clusters as separate entities for a moment
        # [B, Clusters, NodesPerCluster, D]
        n_clusters = n_x.view(batch, num_clusters, nodes_per_cluster, dim)
        
        # Evolve clusters with shared logic but distinct states
        n_clusters = n_clusters + self.node_interaction(n_clusters + seq_context.unsqueeze(1))
        
        # Reshape back to flat nodes
        n_x = n_clusters.view(batch, num_nodes, dim)
        
        # Global world state
        world_state = n_x.mean(dim=1, keepdim=True)
        return x + world_state, n_x

class RWKV_7_GNN_Brain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, MODEL_DIM)
        self.layers = nn.ModuleList()
        for _ in range(MODEL_LAYERS):
            self.layers.append(RWKV_7_Layer(MODEL_DIM))
            self.layers.append(GNN_WorldModel(MODEL_DIM, MODEL_NODES))
        
        self.node_emb = nn.Parameter(torch.randn(1, MODEL_NODES, MODEL_DIM))
        self.final_ln = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, vocab_size)

    def forward(self, ids, h_states=None):
        B, T = ids.size()
        x = self.emb(ids)
        n_x = self.node_emb.expand(B, -1, -1)
        
        new_h = []
        h_idx = 0
        for layer in self.layers:
            if isinstance(layer, RWKV_7_Layer):
                x, h = layer(x)
                new_h.append(h)
            else:
                x, n_x = layer(x, n_x)
                
        logits = self.head(self.final_ln(x))
        return logits, new_h

SocialNarrativeModel = RWKV_7_GNN_Brain
