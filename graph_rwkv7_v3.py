"""
Graph-RWKV-7 V3: Multi-Head WKV + dim=512
==========================================
RWKV-7 native multi-head architecture (NOT transformer attention).

Each head has:
  - Its own [head_dim, head_dim] state matrix
  - Its own graph-derived weight matrices
  - A different VIEW of the graph:
    Head 0: Forward transitions  (what follows what)
    Head 1: Backward transitions (what precedes what)
    Head 2: Semantic PMI         (what co-occurs with what)
    Head 3: Community structure   (what clusters with what)

This is how real RWKV-7 works — parallel recurrent channels, 
NOT transformer self-attention.

dim=512, 4 heads, head_dim=128
State per head: [128, 128] = 16K
Total state: 4 × 16K = 65K
Weight matrices: 6 × [128, 128] × 4 heads = 393K parameters
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from token_graph import TokenGraph


class MultiHeadGraphWKV:
    """
    RWKV-7 Multi-Head WKV with graph-derived weights.
    
    NOT transformer attention. This is RWKV's native multi-head:
      - Each head is an independent recurrent channel
      - Each head has its own state matrix
      - Each head sees a different aspect of the graph
      - Outputs are concatenated (not attended)
    """
    
    def __init__(self, token_graph: TokenGraph, n_heads: int = 4):
        self.graph = token_graph
        self.dim = token_graph.dim
        self.n_heads = n_heads
        self.head_dim = self.dim // n_heads
        self.vocab_size = token_graph.vocab_size
        
        # Embeddings
        self.E = torch.tensor(token_graph.embeddings, dtype=torch.float32)
        
        # Per-head weight matrices (each head sees different graph view)
        self.heads = []
        self._build_heads()
        
        # Output projection: [dim, dim] (learned in real RWKV, we derive from graph)
        self._build_output_projection()
        
        # Frequency log-prior (data distribution bias)
        total = sum(token_graph.token_freq.values()) or 1
        freq_prior = torch.zeros(self.vocab_size)
        for token, idx in token_graph.vocab.items():
            p = token_graph.token_freq[token] / total
            freq_prior[idx] = math.log(p + 1e-10)
        freq_prior = freq_prior - freq_prior.mean()
        freq_prior = freq_prior / (freq_prior.abs().max() + 1e-8)
        self.freq_prior = freq_prior
        
        # Time-mixing
        edge_density = len(token_graph.bigram_freq) / (self.vocab_size ** 2 + 1e-8)
        self.mix_ratio = min(0.8, max(0.3, edge_density * 100))
        
        # Per-head state [head_dim, head_dim] and prev embedding
        self.states = [torch.zeros(self.head_dim, self.head_dim) for _ in range(n_heads)]
        self.x_prev = torch.zeros(self.dim)
        self.step_count = 0
    
    def _build_heads(self):
        """
        Build weight matrices for each head from different graph views.
        
        Head 0: Forward transitions  — what token follows what
        Head 1: Backward transitions — what token precedes what  
        Head 2: Semantic PMI         — what co-occurs with what
        Head 3: Community structure  — what clusters together
        """
        E = self.graph.embeddings  # [V, dim]
        V = self.vocab_size
        hd = self.head_dim
        
        # Prepare graph matrices
        T_fwd = self.graph.transition                    # Forward transitions
        T_bwd = self.graph.transition.T.copy()           # Backward transitions
        # Re-normalize backward
        bwd_sums = T_bwd.sum(axis=1, keepdims=True)
        bwd_sums = np.where(bwd_sums == 0, 1, bwd_sums)
        T_bwd = T_bwd / bwd_sums
        
        PMI = self.graph.pmi_matrix / (self.graph.pmi_matrix.max() + 1e-8)
        
        # Community similarity
        cluster_ids = np.zeros(V, dtype=np.int32)
        for token, cid in self.graph.community_map.items():
            if token in self.graph.vocab:
                cluster_ids[self.graph.vocab[token]] = cid
        n_clusters = max(cluster_ids) + 1 if len(cluster_ids) > 0 else 1
        C = np.zeros((V, n_clusters), dtype=np.float32)
        for i in range(V):
            C[i, cluster_ids[i]] = 1.0
        cluster_sim = C @ C.T
        cluster_sim = cluster_sim / (cluster_sim.sum(axis=1, keepdims=True) + 1e-8)
        
        # PageRank
        pr = self.graph.pagerank_scores / (self.graph.pagerank_scores.max() + 1e-8)
        pr_diag = np.diag(pr)
        
        # Log-adjacency
        log_adj = np.log1p(self.graph.adjacency)
        log_adj = log_adj / (log_adj.max() + 1e-8)
        
        # Frequency gating
        max_freq = max(self.graph.token_freq.values()) if self.graph.token_freq else 1
        log_freq = np.zeros(V, dtype=np.float32)
        for token, idx in self.graph.vocab.items():
            log_freq[idx] = np.log1p(self.graph.token_freq[token]) / np.log1p(max_freq)
        gate_diag = np.diag(1.0 - log_freq)
        
        # Define graph views for each head
        graph_views = [
            {  # Head 0: Forward sequential patterns
                'r_source': pr_diag,
                'w_source': log_adj,
                'k_source': T_fwd,
                'v_source': PMI,
                'a_source': cluster_sim,
                'g_source': gate_diag,
            },
            {  # Head 1: Backward sequential patterns  
                'r_source': pr_diag,
                'w_source': log_adj.T,
                'k_source': T_bwd,
                'v_source': PMI.T,
                'a_source': cluster_sim,
                'g_source': gate_diag,
            },
            {  # Head 2: Semantic (PMI-driven)
                'r_source': PMI,
                'w_source': cluster_sim,
                'k_source': PMI,
                'v_source': T_fwd,
                'a_source': pr_diag,
                'g_source': gate_diag,
            },
            {  # Head 3: Structural (community-driven)
                'r_source': cluster_sim,
                'w_source': gate_diag,
                'k_source': cluster_sim,
                'v_source': PMI,
                'a_source': T_fwd,
                'g_source': pr_diag,
            },
        ]
        
        # Use only first n_heads views
        for h in range(self.n_heads):
            view = graph_views[h % len(graph_views)]
            
            # Per-head INPUT projection: [dim] -> [head_dim]
            # Each head projects the FULL embedding through a different graph matrix
            # This is key: every head sees ALL token information, not just a slice
            M_proj = view['k_source']  # Use the head's primary graph view for projection
            E_proj = E.T @ M_proj @ E  # [dim, dim]
            # SVD to get a clean [dim, head_dim] projection
            U, S, Vt = np.linalg.svd(E_proj, full_matrices=False)
            head_proj = U[:, :hd] * np.sqrt(S[:hd] + 1e-8)[np.newaxis, :]  # [dim, head_dim]
            # Normalize
            hp_norm = np.linalg.norm(head_proj, axis=0, keepdims=True)
            hp_norm = np.where(hp_norm == 0, 1, hp_norm)
            head_proj = head_proj / hp_norm
            
            # Projected embeddings for this head: [V, head_dim]
            E_head = E @ head_proj  # Each token's full info projected to head space
            # Normalize
            eh_norm = np.linalg.norm(E_head, axis=1, keepdims=True)
            eh_norm = np.where(eh_norm == 0, 1, eh_norm)
            E_head = E_head / eh_norm
            
            head_weights = {}
            head_weights['proj'] = torch.tensor(head_proj, dtype=torch.float32)  # [dim, head_dim]
            scale = np.sqrt(2.0 / hd)
            
            for name in ['r', 'w', 'k', 'v', 'a', 'g']:
                M = view[f'{name}_source']
                W = E_head.T @ M @ E_head  # [head_dim, head_dim]
                
                # Add identity component for stability
                if name == 'r':
                    W = W + np.eye(hd) * 0.5
                elif name == 'w':
                    W = W * 0.5 - np.eye(hd) * 1.5  # Bias toward decay
                elif name in ('k', 'v'):
                    W = W + np.eye(hd) * 0.1
                elif name in ('a', 'g'):
                    W = W * 0.3
                
                # Xavier scale
                w_max = np.abs(W).max()
                if w_max > 0:
                    W = W * scale / w_max
                
                head_weights[f'W_{name}'] = torch.tensor(W, dtype=torch.float32)
            
            self.heads.append(head_weights)
    
    def _build_output_projection(self):
        """Output projection from graph structure."""
        E = self.graph.embeddings
        # Mix forward+backward transitions for output
        M = self.graph.transition + self.graph.transition.T
        M = M / (M.max() + 1e-8)
        W_out = E.T @ M @ E
        scale = np.sqrt(2.0 / self.dim)
        w_max = np.abs(W_out).max()
        if w_max > 0:
            W_out = W_out * scale / w_max
        self.W_out = torch.tensor(W_out + np.eye(self.dim) * 0.3, dtype=torch.float32)
        
    def load_trained_weights(self, state_dict):
        """Load fine-tuned weights from TrainableGraphRWKV."""
        self.E = state_dict['E'].detach().cpu()
        self.W_out = state_dict['W_out'].detach().cpu()
        self.mix_ratio = state_dict['mix_ratio'].item()
        
        for h in range(self.n_heads):
            self.heads[h]['proj'] = state_dict[f'head_projs.{h}'].detach().cpu()
            self.heads[h]['W_r'] = state_dict[f'head_W_r.{h}'].detach().cpu()
            self.heads[h]['W_w'] = state_dict[f'head_W_w.{h}'].detach().cpu()
            self.heads[h]['W_k'] = state_dict[f'head_W_k.{h}'].detach().cpu()
            self.heads[h]['W_v'] = state_dict[f'head_W_v.{h}'].detach().cpu()
            self.heads[h]['W_a'] = state_dict[f'head_W_a.{h}'].detach().cpu()
            self.heads[h]['W_g'] = state_dict[f'head_W_g.{h}'].detach().cpu()
            
    def step(self, token_id: int) -> torch.Tensor:
        """
        Multi-head WKV step.
        
        Each head independently:
          1. Projects x_mixed through its graph-derived weights
          2. Updates its state via Generalized Delta Rule
          3. Produces a head_dim output
        
        Outputs are concatenated → [dim] → output projection.
        """
        x = self.E[token_id] if 0 <= token_id < self.vocab_size else torch.zeros(self.dim)
        
        # Time-mixing
        x_mixed = x * self.mix_ratio + self.x_prev * (1 - self.mix_ratio)
        
        head_outputs = []
        
        for h in range(self.n_heads):
            hd = self.head_dim
            hs = h * hd
            he = hs + hd
            
            # Project full input to head space using per-head projection
            x_h = self.heads[h]['proj'].T @ x_mixed  # [head_dim]
            
            # Get this head's weight matrices
            W = self.heads[h]
            
            # Context-dependent projections
            r = torch.sigmoid(W['W_r'] @ x_h)
            w = torch.exp(torch.clamp(W['W_w'] @ x_h, -3.0, 0.0))
            k = W['W_k'] @ x_h
            v = W['W_v'] @ x_h
            a = torch.sigmoid(W['W_a'] @ x_h)
            g = torch.sigmoid(W['W_g'] @ x_h)
            
            # Generalized Delta Rule
            r_col = r.unsqueeze(-1)
            k_row = k.unsqueeze(0)
            v_col = v.unsqueeze(-1)
            a_col = a.unsqueeze(-1)
            g_col = g.unsqueeze(-1)
            
            state = self.states[h]
            state = state * w.unsqueeze(0)
            state = state - (state @ (g_col * a_col)) @ g_col.T
            state = state + (v_col @ k_row) * 0.15
            
            # Normalize state
            sn = state.norm()
            if sn > 5.0:
                state = state * (5.0 / sn)
            
            self.states[h] = state
            
            # Head output
            out_h = (state @ r_col).squeeze(-1)  # [head_dim]
            head_outputs.append(out_h)
        
        # Concatenate all heads: [dim]
        output = torch.cat(head_outputs, dim=0)
        
        # Output projection
        output = self.W_out @ output
        
        self.x_prev = x.clone()
        self.step_count += 1
        
        return output
    
    def compute_logits(self, output: torch.Tensor, last_token_id: int = -1) -> torch.Tensor:
        """
        Logits = blend of three signals:
          1. WKV state similarity (context-dependent, from multi-head state)
          2. Frequency prior (unigram data distribution)
          3. Transition distribution (P(next|current) from graph)
        
        The transition distribution is NOT a bigram hack. In trained RWKV-7,
        the output head (lm_head) learns token-to-token transition patterns.
        We're using the graph's transition matrix as that output head.
        """
        out_norm = output / (output.norm() + 1e-8)
        emb_norm = self.E / (self.E.norm(dim=1, keepdim=True) + 1e-8)
        
        # Signal 1: WKV state similarity
        wkv_logits = emb_norm @ out_norm * 10.0
        
        # Signal 2: Frequency prior
        prior_logits = self.freq_prior * 8.0
        
        # Signal 3: Transition distribution (graph's "output head")
        trans_logits = torch.zeros(self.vocab_size)
        if 0 <= last_token_id < self.vocab_size:
            trans_row = self.graph.transition[last_token_id]  # [V]
            trans_logits = torch.tensor(trans_row, dtype=torch.float32)
            # Log-scale for logit space
            trans_logits = torch.log(trans_logits + 1e-10)
            trans_logits = trans_logits - trans_logits.mean()
            trans_logits = trans_logits / (trans_logits.abs().max() + 1e-8) * 10.0
        
        # Blend: WKV dominates as context grows
        wkv_w = min(0.6, 0.3 + self.step_count * 0.005)
        trans_w = 0.3  # Transition distribution is always useful
        prior_w = 1.0 - wkv_w - trans_w
        
        return wkv_logits * wkv_w + trans_logits * trans_w + prior_logits * prior_w
    
    def soft_reset(self, decay: float = 0.1):
        """Soft reset between conversations."""
        for h in range(self.n_heads):
            self.states[h] = self.states[h] * decay
        self.x_prev = self.x_prev * decay


class InfiniteContextGeneratorV3:
    """
    Text generator with:
      - dim=512 (262K parameter budget)
      - 4-head WKV (each head sees different graph view)
      - Infinite context session
      - Zero rules, zero templates
    """
    
    def __init__(self, token_graph: TokenGraph, n_heads: int = 4):
        self.graph = token_graph
        self.rwkv = MultiHeadGraphWKV(token_graph, n_heads)
        self.vocab = token_graph.vocab
        self.id_to_token = token_graph.id_to_token
        self.vocab_size = token_graph.vocab_size
        self.session_tokens = []
    
    def generate(self, prompt: str = "", max_tokens: int = 60,
                 temperature: float = 0.7, top_k: int = 20,
                 top_p: float = 0.9, repetition_penalty: float = 1.3,
                 seed: Optional[int] = None) -> str:
        """Generate with infinite context. No rules. No templates."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if prompt:
            prompt_tokens = TokenGraph.tokenize(prompt)
        else:
            top = self.graph.top_tokens(20)
            content = [t for t, s in top if len(t) > 2]
            prompt_tokens = [content[0]] if content else ["the"]
        
        # Feed prompt through WKV
        for token in prompt_tokens:
            if token in self.vocab:
                self.rwkv.step(self.vocab[token])
                self.session_tokens.append(token)
        
        generated = list(prompt_tokens)
        
        for _ in range(max_tokens):
            current = generated[-1] if generated else ""
            cid = self.vocab.get(current, 0)
            
            output = self.rwkv.step(cid)
            logits = self.rwkv.compute_logits(output, last_token_id=cid)
            
            # Repetition penalty
            recent = self.session_tokens[-20:] + generated[-20:]
            for t in set(recent):
                if t in self.vocab:
                    count = recent.count(t)
                    logits[self.vocab[t]] /= repetition_penalty ** min(count, 3)
            
            next_id = self._sample(logits, temperature, top_k, top_p)
            next_token = self.id_to_token.get(next_id, "")
            if not next_token:
                continue
            
            generated.append(next_token)
            self.session_tokens.append(next_token)
            
            if next_token in {'.', '!', '?'} and len(generated) > len(prompt_tokens) + 8:
                break
        
        return self._detokenize(generated)
    
    def _sample(self, logits, temperature, top_k, top_p):
        logits = logits / max(temperature, 1e-8)
        if 0 < top_k < len(logits):
            tv, _ = torch.topk(logits, top_k)
            logits = torch.where(logits >= tv[-1], logits, torch.full_like(logits, -1e9))
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            mask = (cum - sp) > top_p
            sp[mask] = 0.0
            sp = sp / (sp.sum() + 1e-8)
            return si[torch.multinomial(sp, 1).item()].item()
        return torch.multinomial(probs, 1).item()
    
    @staticmethod
    def _detokenize(tokens):
        # BPE tokens from tiktoken already contain the correct whitespace (e.g. ' hello', 'world')
        # We just need to join them directly.
        if not tokens:
            return ""
        return "".join(tokens)
    
    def session_stats(self):
        return {
            'session_tokens': len(self.session_tokens),
            'state_norms': [s.norm().item() for s in self.rwkv.states],
            'step_count': self.rwkv.step_count,
            'n_heads': self.rwkv.n_heads,
            'head_dim': self.rwkv.head_dim,
            'total_dim': self.rwkv.dim,
        }
