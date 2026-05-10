"""
Token Graph V2: Pure Statistical Graph — Zero Hardcoded Rules
=============================================================
Replaces word_graph.py. No POS, no grammar, no language-specific rules.

Works on ANY sequential data:
  - English text
  - Hindi/Hinglish text  
  - Python/C++/JavaScript code
  - Mixed data

Graph captures ALL patterns purely from token statistics:
  - Token transitions (bigrams) → Adjacency matrix
  - Co-occurrence (window) → PMI matrix
  - Spectral decomposition → Token embeddings
  - Community structure → Cluster similarity matrix
"""

import re
import math
import json
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import tiktoken

import numpy as np
import networkx as nx


class TokenGraph:
    """
    Pure statistical token graph. No rules. No heuristics.
    
    Learns structure from data:
      - Transition probabilities (what follows what)
      - Co-occurrence patterns (what appears near what)
      - Spectral structure (graph eigenvectors = embeddings)
      - Communities (graph clustering)
    """
    
    def __init__(self, dim: int = 128, window_size: int = 5):
        self.dim = dim
        self.window_size = window_size
        
        # Token statistics
        self.token_freq = Counter()
        self.bigram_freq = Counter()
        self.cooccurrence = defaultdict(Counter)
        self.total_tokens = 0
        
        # Vocab
        self.vocab = {}       # token → id
        self.id_to_token = {} # id → token
        self.vocab_size = 0
        
        # Matrices (computed after build)
        self.adjacency = None       # [V, V] transition counts
        self.transition = None      # [V, V] transition probabilities  
        self.pmi_matrix = None      # [V, V] pointwise mutual information
        self.embeddings = None      # [V, dim] spectral embeddings
        self.pagerank_scores = None # [V] importance scores
        self.community_map = None   # token → cluster_id
        self.num_communities = 0
        
        # Graph-derived weight matrices for RWKV-7
        self.W_r = None  # [dim, dim] Receptance weights
        self.W_w = None  # [dim, dim] Decay weights
        self.W_k = None  # [dim, dim] Key weights
        self.W_v = None  # [dim, dim] Value weights
        self.W_a = None  # [dim, dim] Learning rate weights
        self.W_g = None  # [dim, dim] Gate weights
        
        # Cache
        self._hashes = {}
    
    _tokenizer = None
    
    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = tiktoken.get_encoding("cl100k_base")
        return cls._tokenizer
        
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Universal Sub-word Tokenizer (BPE).
        Uses OpenAI's tiktoken (cl100k_base) to split text into word + character pieces.
        This provides morphological context (e.g., 'ghoomne' -> 'ghoom' + 'ne').
        """
        tokenizer = cls.get_tokenizer()
        # Encode to BPE token IDs
        token_ids = tokenizer.encode(text)
        # Convert each ID back to a string token (with replacement for partial bytes)
        return [tokenizer.decode_bytes([t]).decode('utf-8', errors='replace') for t in token_ids]
    
    # =========================================================================
    # DATA INGESTION
    # =========================================================================
    
    def add_text(self, text: str, source_id: str = "data") -> bool:
        """Add text data. Returns True if new content processed."""
        h = hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()
        if source_id in self._hashes and self._hashes[source_id] == h:
            return False
        self._hashes[source_id] = h
        
        tokens = self.tokenize(text)
        if not tokens:
            return True
        
        self.total_tokens += len(tokens)
        
        # Count frequencies
        for t in tokens:
            self.token_freq[t] += 1
        
        # Bigrams
        for i in range(len(tokens) - 1):
            self.bigram_freq[(tokens[i], tokens[i+1])] += 1
        
        # Co-occurrence within window
        for i in range(len(tokens)):
            start = max(0, i - self.window_size)
            end = min(len(tokens), i + self.window_size + 1)
            for j in range(start, end):
                if i != j:
                    self.cooccurrence[tokens[i]][tokens[j]] += 1
        
        return True
    
    def add_texts(self, texts: List[str]):
        """Add multiple texts."""
        for i, text in enumerate(texts):
            self.add_text(text, source_id=f"text_{i}")
    
    # =========================================================================
    # BUILD — Construct all matrices from statistics
    # =========================================================================
    
    def build(self, min_freq: int = 1):
        """
        Build everything from token statistics.
        No rules applied — pure math on the graph.
        """
        # 1. Build vocab (frequency-filtered)
        self._build_vocab(min_freq)
        
        # 2. Build adjacency/transition matrices
        self._build_adjacency()
        
        # 3. Compute PMI matrix (semantic similarity)
        self._build_pmi()
        
        # 4. Spectral embedding (graph eigenvectors)
        self._build_spectral_embedding()
        
        # 5. PageRank (node importance)
        self._build_pagerank()
        
        # 6. Community detection
        self._build_communities()
        
        # 7. Construct RWKV-7 weight matrices from graph
        self._build_weight_matrices()
        
        return self
    
    def _build_vocab(self, min_freq: int):
        """Build vocabulary from frequency counts."""
        self.vocab = {}
        self.id_to_token = {}
        
        for token, freq in self.token_freq.most_common():
            if freq >= min_freq:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id_to_token[idx] = token
        
        self.vocab_size = len(self.vocab)
    
    def _build_adjacency(self):
        """Build adjacency and transition matrices from bigram counts."""
        V = self.vocab_size
        self.adjacency = np.zeros((V, V), dtype=np.float32)
        
        for (t1, t2), freq in self.bigram_freq.items():
            if t1 in self.vocab and t2 in self.vocab:
                i, j = self.vocab[t1], self.vocab[t2]
                self.adjacency[i, j] = freq
        
        # Row-normalize for transition probabilities
        row_sums = self.adjacency.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition = self.adjacency / row_sums
    
    def _build_pmi(self):
        """Build Pointwise Mutual Information matrix (semantic relationships)."""
        V = self.vocab_size
        self.pmi_matrix = np.zeros((V, V), dtype=np.float32)
        
        total = self.total_tokens
        if total == 0:
            return
        
        for t1, neighbors in self.cooccurrence.items():
            if t1 not in self.vocab:
                continue
            i = self.vocab[t1]
            p_t1 = self.token_freq[t1] / total
            
            for t2, cofreq in neighbors.items():
                if t2 not in self.vocab:
                    continue
                j = self.vocab[t2]
                p_t2 = self.token_freq[t2] / total
                p_joint = cofreq / total
                
                if p_t1 > 0 and p_t2 > 0 and p_joint > 0:
                    pmi = math.log2(p_joint / (p_t1 * p_t2))
                    self.pmi_matrix[i, j] = max(pmi, 0)  # Positive PMI only
    
    def _build_spectral_embedding(self):
        """
        Spectral embedding from graph structure.
        
        Uses frequency-weighted adjacency so common tokens get
        proportional representation (prevents rare-token domination).
        
        No handcrafted features. Pure graph math.
        """
        V = self.vocab_size
        
        # Symmetric adjacency (undirected version for spectral)
        A_sym = self.adjacency + self.adjacency.T
        
        # Add PMI as semantic edges (scaled to not dominate)
        pmi_scale = A_sym.max() / (self.pmi_matrix.max() + 1e-8) * 0.1
        A_combined = A_sym + self.pmi_matrix * pmi_scale
        
        # Frequency weighting: scale rows/cols by sqrt(freq)
        # This ensures common tokens have stronger spectral presence
        max_freq = max(self.token_freq.values()) if self.token_freq else 1
        freq_weights = np.zeros(V, dtype=np.float32)
        for token, idx in self.vocab.items():
            freq_weights[idx] = np.sqrt(self.token_freq[token] / max_freq + 0.01)
        F_diag = np.diag(freq_weights)
        A_weighted = F_diag @ A_combined @ F_diag
        
        # Degree matrix for normalization
        degrees = A_weighted.sum(axis=1)
        degrees = np.where(degrees == 0, 1, degrees)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        
        # Normalized Laplacian-style
        A_norm = D_inv_sqrt @ A_weighted @ D_inv_sqrt
        
        # Eigendecomposition
        actual_dim = min(self.dim, V - 1) if V > 1 else 1
        
        try:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            A_sparse = csr_matrix(A_norm)
            eigenvalues, eigenvectors = eigsh(A_sparse, k=actual_dim, which='LM')
        except (ImportError, Exception):
            eigenvalues, eigenvectors_full = np.linalg.eigh(A_norm)
            idx = np.argsort(eigenvalues)[::-1][:actual_dim]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors_full[:, idx]
        
        # Scale by sqrt(eigenvalue) — higher eigenvalues = more important dimensions
        scale = np.sqrt(np.abs(eigenvalues) + 1e-8)
        embeddings = eigenvectors * scale[np.newaxis, :]
        
        # Pad to full dim if needed
        if embeddings.shape[1] < self.dim:
            pad_width = self.dim - embeddings.shape[1]
            pad = np.random.randn(V, pad_width).astype(np.float32) * 0.01
            embeddings = np.concatenate([embeddings, pad], axis=1)
        
        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = (embeddings / norms).astype(np.float32)
    
    def _build_pagerank(self):
        """Compute PageRank on the token graph."""
        G = nx.DiGraph()
        for (t1, t2), freq in self.bigram_freq.items():
            if t1 in self.vocab and t2 in self.vocab:
                G.add_edge(t1, t2, weight=freq)
        
        if len(G) == 0:
            self.pagerank_scores = np.ones(self.vocab_size) / self.vocab_size
            return
        
        try:
            pr = nx.pagerank(G, alpha=0.85, weight='weight')
        except nx.PowerIterationFailedConvergence:
            pr = {n: 1.0 / len(G) for n in G.nodes()}
        
        self.pagerank_scores = np.zeros(self.vocab_size, dtype=np.float32)
        for token, score in pr.items():
            if token in self.vocab:
                self.pagerank_scores[self.vocab[token]] = score
    
    def _build_communities(self):
        """Detect communities (no language rules — pure graph clustering)."""
        G = nx.Graph()
        for (t1, t2), freq in self.bigram_freq.items():
            if t1 in self.vocab and t2 in self.vocab and freq >= 2:
                G.add_edge(t1, t2, weight=freq)
        
        if len(G) < 2:
            self.community_map = {t: 0 for t in self.vocab}
            self.num_communities = 1
            return
        
        try:
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
        except Exception:
            communities = [{n} for n in G.nodes()]
        
        self.community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                self.community_map[node] = i
        
        # Assign unclustered tokens to community 0
        for token in self.vocab:
            if token not in self.community_map:
                self.community_map[token] = 0
        
        self.num_communities = len(communities)
    
    def _build_weight_matrices(self):
        """
        THE KEY FUNCTION: Construct RWKV-7 weight matrices from graph.
        
        W_r = E^T @ diag(pagerank) @ E          — Receptance
        W_w = E^T @ decay_matrix @ E             — Decay  
        W_k = E^T @ transition_matrix @ E        — Key
        W_v = E^T @ pmi_matrix @ E               — Value
        W_a = E^T @ cluster_similarity @ E       — Learning rate
        W_g = E^T @ gate_matrix @ E              — Gate
        
        All [dim, dim]. All produce context-dependent outputs via W @ x.
        """
        E = self.embeddings  # [V, dim]
        V = self.vocab_size
        
        # --- W_r: Receptance (from PageRank — how much to accept) ---
        # PageRank-weighted projection + identity for stability
        pr_diag = np.diag(self.pagerank_scores / (self.pagerank_scores.max() + 1e-8))
        W_r_graph = E.T @ pr_diag @ E
        self.W_r = (W_r_graph + np.eye(self.dim) * 0.5).astype(np.float32)
        
        # --- W_w: Decay (from transition strength) ---
        # Strong transitions = slow decay (high w = remember)
        # Use log-transition to compress range
        log_trans = np.log1p(self.adjacency)
        log_trans = log_trans / (log_trans.max() + 1e-8)
        W_w_graph = E.T @ log_trans @ E
        # Bias toward negative values so exp(W_w @ x) < 1 (decay)
        self.W_w = (W_w_graph * 0.5 - np.eye(self.dim) * 1.5).astype(np.float32)
        
        # --- W_k: Key (from transition probabilities) ---
        W_k_graph = E.T @ self.transition @ E
        self.W_k = (W_k_graph + np.eye(self.dim) * 0.1).astype(np.float32)
        
        # --- W_v: Value (from PMI — semantic co-occurrence) ---
        pmi_norm = self.pmi_matrix / (self.pmi_matrix.max() + 1e-8)
        W_v_graph = E.T @ pmi_norm @ E
        self.W_v = (W_v_graph + np.eye(self.dim) * 0.3).astype(np.float32)
        
        # --- W_a: Learning rate (from community structure) ---
        # Build cluster similarity efficiently
        cluster_ids = np.zeros(V, dtype=np.int32)
        for token, cid in self.community_map.items():
            if token in self.vocab:
                cluster_ids[self.vocab[token]] = cid
        # One-hot cluster membership
        n_clusters = max(cluster_ids) + 1 if len(cluster_ids) > 0 else 1
        cluster_onehot = np.zeros((V, n_clusters), dtype=np.float32)
        for i in range(V):
            cluster_onehot[i, cluster_ids[i]] = 1.0
        # Cluster similarity = C @ C^T (dot product of one-hot = 1 if same cluster)
        cluster_sim = cluster_onehot @ cluster_onehot.T
        cluster_sim = cluster_sim / (cluster_sim.sum(axis=1, keepdims=True) + 1e-8)
        W_a_graph = E.T @ cluster_sim @ E
        self.W_a = (W_a_graph * 0.3).astype(np.float32)
        
        # --- W_g: Gate (frequency-aware gating) ---
        max_freq = max(self.token_freq.values()) if self.token_freq else 1
        # Log-frequency based (compressed range)
        log_freq = np.zeros(V, dtype=np.float32)
        for token, idx in self.vocab.items():
            log_freq[idx] = np.log1p(self.token_freq[token]) / np.log1p(max_freq)
        gate_matrix = np.diag(1.0 - log_freq)  # Rare = higher gate
        W_g_graph = E.T @ gate_matrix @ E
        self.W_g = (W_g_graph * 0.3).astype(np.float32)
        
        # Xavier-scale all weight matrices
        scale = np.sqrt(2.0 / self.dim)
        for name in ['W_r', 'W_w', 'W_k', 'W_v', 'W_a', 'W_g']:
            W = getattr(self, name)
            w_max = np.abs(W).max()
            if w_max > 0:
                setattr(self, name, (W * scale / w_max).astype(np.float32))
    
    # =========================================================================
    # INFO
    # =========================================================================
    
    def stats(self) -> dict:
        """Return graph statistics."""
        return {
            'vocab_size': self.vocab_size,
            'total_tokens': self.total_tokens,
            'unique_bigrams': len(self.bigram_freq),
            'communities': self.num_communities,
            'dim': self.dim,
            'embedding_shape': self.embeddings.shape if self.embeddings is not None else None,
        }
    
    def top_tokens(self, k: int = 10) -> List[Tuple[str, float]]:
        """Top tokens by PageRank."""
        if self.pagerank_scores is None:
            return []
        indices = np.argsort(self.pagerank_scores)[::-1][:k]
        return [(self.id_to_token[i], float(self.pagerank_scores[i])) for i in indices]
    
    def save(self, path: str):
        """Save graph to file."""
        data = {
            'vocab': self.vocab,
            'token_freq': dict(self.token_freq),
            'bigram_freq': {f"{k[0]}|||{k[1]}": v for k, v in self.bigram_freq.items()},
            'dim': self.dim,
            'window_size': self.window_size,
            'total_tokens': self.total_tokens,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'TokenGraph':
        """Load graph from file and rebuild matrices."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph = cls(dim=data['dim'], window_size=data['window_size'])
        graph.vocab = data['vocab']
        graph.id_to_token = {int(v): k for k, v in data['vocab'].items()}
        graph.token_freq = Counter(data['token_freq'])
        graph.total_tokens = data['total_tokens']
        
        for key, freq in data['bigram_freq'].items():
            t1, t2 = key.split('|||')
            graph.bigram_freq[(t1, t2)] = freq
        
        graph.build(min_freq=1)
        return graph
