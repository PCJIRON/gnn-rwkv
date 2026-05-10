"""
Graph-RWKV-7: Training-Free Text Generation
============================================
Uses the word graph built by Graphify-NLP to directly compute RWKV-7 WKV
state values WITHOUT any gradient-based training.

Key Insight:
  Graph structure (PageRank, degree, co-occurrence, communities) 
  provides the R, W, K, V, A, G values that RWKV-7 normally learns.

  - R (Receptance) ← PageRank score (how much to accept)
  - W (Decay) ← exp(-1 / (degree + 1)) (how fast to forget)
  - K (Key) ← Graph-derived word embedding
  - V (Value) ← Neighborhood mean embedding
  - A (Learning Rate) ← Cluster similarity  
  - G (Gate) ← Co-occurrence strength
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import networkx as nx

from word_graph import WordGraphBuilder


class GraphEmbedding:
    """
    Compute word embeddings directly from graph structure.
    No training needed — uses spectral/structural features.
    """
    
    def __init__(self, graph_builder: WordGraphBuilder, dim: int = 128):
        self.builder = graph_builder
        self.graph = graph_builder.graph
        self.dim = dim
        self.vocab = graph_builder.vocab
        self.id_to_word = graph_builder.id_to_word
        self.vocab_size = len(self.vocab)
        
        # Computed embeddings
        self.embeddings = None  # [vocab_size, dim]
        self._pagerank = {}
        self._communities = {}
        self._degrees = {}
    
    def compute_all(self):
        """Compute all graph-derived features."""
        self._pagerank = self.builder.compute_pagerank()
        self._communities = self.builder.detect_communities()
        
        # Degree of word nodes
        word_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "WORD"]
        for node in word_nodes:
            self._degrees[node] = self.graph.degree(node)
        
        # Build embeddings from graph features
        self._build_embeddings()
    
    def _build_embeddings(self):
        """
        Build word embeddings from graph structure (no training).
        
        Features for each word:
        1. Normalized frequency (log scale)
        2. PageRank score
        3. Community one-hot (or hash)
        4. Degree centrality
        5. POS one-hot
        6. Neighbor statistics
        7. Random walk features (DeepWalk-lite)
        """
        embeddings = torch.zeros(self.vocab_size, self.dim)
        
        # Feature allocation
        freq_dim = 8
        pr_dim = 8
        comm_dim = 16
        degree_dim = 8
        pos_dim = 16
        neighbor_dim = 32
        walk_dim = self.dim - freq_dim - pr_dim - comm_dim - degree_dim - pos_dim - neighbor_dim
        
        pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "DET", "PREP", "CONJ", "PRON", "PUNCT"]
        pos_to_idx = {p: i for i, p in enumerate(pos_tags)}
        
        max_freq = max(self.builder.word_freq.values()) if self.builder.word_freq else 1
        max_degree = max(self._degrees.values()) if self._degrees else 1
        max_pr = max(self._pagerank.values()) if self._pagerank else 1
        
        for word, idx in self.vocab.items():
            offset = 0
            
            # 1. Frequency features (spread across freq_dim)
            freq = self.builder.word_freq.get(word, 0)
            log_freq = math.log1p(freq) / math.log1p(max_freq)
            for d in range(freq_dim):
                embeddings[idx, offset + d] = log_freq * math.sin((d + 1) * math.pi * log_freq)
            offset += freq_dim
            
            # 2. PageRank features
            pr = self._pagerank.get(word, 0)
            pr_norm = pr / (max_pr + 1e-8)
            for d in range(pr_dim):
                embeddings[idx, offset + d] = pr_norm * math.cos((d + 1) * math.pi * pr_norm)
            offset += pr_dim
            
            # 3. Community features (hash-based encoding)
            comm = self._communities.get(word, 0)
            for d in range(comm_dim):
                embeddings[idx, offset + d] = math.sin((comm + 1) * (d + 1) * 0.7)
            offset += comm_dim
            
            # 4. Degree features
            deg = self._degrees.get(word, 0)
            deg_norm = deg / (max_degree + 1e-8)
            for d in range(degree_dim):
                embeddings[idx, offset + d] = deg_norm * math.sin((d + 1) * 2.0 * deg_norm)
            offset += degree_dim
            
            # 5. POS features (one-hot style with smoothing)
            pos = self.builder.word_to_pos.get(word, "NOUN")
            pos_idx = pos_to_idx.get(pos, 0)
            for d in range(pos_dim):
                if d < len(pos_tags):
                    embeddings[idx, offset + d] = 1.0 if d == pos_idx else 0.05
                else:
                    embeddings[idx, offset + d] = 0.1 * math.sin(pos_idx * d)
            offset += pos_dim
            
            # 6. Neighbor features (aggregate neighbor statistics)
            if word in self.graph:
                neighbors = list(self.graph.successors(word))
                word_neighbors = [n for n in neighbors 
                                  if self.graph.nodes[n].get("type") == "WORD"]
                
                if word_neighbors:
                    n_freqs = [self.builder.word_freq.get(n, 0) for n in word_neighbors]
                    n_prs = [self._pagerank.get(n, 0) for n in word_neighbors]
                    
                    stats = [
                        len(word_neighbors) / (max_degree + 1e-8),  # normalized neighbor count
                        np.mean(n_freqs) / (max_freq + 1e-8),
                        np.std(n_freqs) / (max_freq + 1e-8) if len(n_freqs) > 1 else 0,
                        np.mean(n_prs) / (max_pr + 1e-8),
                        np.max(n_prs) / (max_pr + 1e-8) if n_prs else 0,
                    ]
                    
                    for d in range(neighbor_dim):
                        s_idx = d % len(stats)
                        embeddings[idx, offset + d] = stats[s_idx] * math.sin((d + 1) * 0.5)
            offset += neighbor_dim
            
            # 7. Random walk features (simplified DeepWalk)
            walk_features = self._random_walk_features(word, walk_dim)
            embeddings[idx, offset:offset + walk_dim] = torch.tensor(walk_features)
        
        # Normalize embeddings
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.embeddings = embeddings / norms
    
    def _random_walk_features(self, word: str, dim: int, num_walks: int = 10, 
                               walk_length: int = 5) -> List[float]:
        """Simplified DeepWalk — random walk based features."""
        features = [0.0] * dim
        
        if word not in self.graph:
            return features
        
        visit_counts = defaultdict(int)
        
        for _ in range(num_walks):
            current = word
            for step in range(walk_length):
                neighbors = [n for n in self.graph.successors(current)
                            if self.graph.nodes.get(n, {}).get("type") == "WORD"]
                if not neighbors:
                    break
                
                # Weighted random choice
                weights = []
                for n in neighbors:
                    w = self.graph[current][n].get("weight", 1.0)
                    weights.append(w)
                
                total_w = sum(weights)
                if total_w == 0:
                    break
                
                probs = [w / total_w for w in weights]
                chosen_idx = np.random.choice(len(neighbors), p=probs)
                current = neighbors[chosen_idx]
                visit_counts[current] += 1
        
        # Convert visit counts to features
        sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (visited_word, count) in enumerate(sorted_visits[:dim]):
            features[i] = count / (num_walks * walk_length)
        
        return features
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding for a single word."""
        if word in self.vocab and self.embeddings is not None:
            return self.embeddings[self.vocab[word]]
        return None
    
    def get_embedding_by_id(self, idx: int) -> Optional[torch.Tensor]:
        """Get embedding by vocab index."""
        if self.embeddings is not None and 0 <= idx < self.vocab_size:
            return self.embeddings[idx]
        return None


class GraphRWKV7:
    """
    RWKV-7 WKV operator with graph-derived state values.
    NO TRAINING — all R, W, K, V, A, G computed from graph structure.
    
    State update (Generalized Delta Rule):
      S_t = diag(w_t) * S_{t-1} - (S @ (g_t * a_t)) @ g_t^T + v_t^T @ k_t
      output_t = S_t @ r_t
    """
    
    def __init__(self, graph_builder: WordGraphBuilder, dim: int = 128):
        self.builder = graph_builder
        self.dim = dim
        self.graph_emb = GraphEmbedding(graph_builder, dim)
        self.graph_emb.compute_all()
        
        self.vocab = graph_builder.vocab
        self.id_to_word = graph_builder.id_to_word
        self.vocab_size = len(self.vocab)
        
        # RWKV state matrix: [dim, dim]
        self.state = torch.zeros(dim, dim)
        
        # Grammar model for filtering
        self.grammar_model = graph_builder.get_grammar_model()
        
        # Precompute per-word WKV values
        self._precompute_wkv_values()
    
    def _precompute_wkv_values(self):
        """
        Precompute R, W, K, V, A, G for every word in vocab.
        These are normally LEARNED by RWKV-7 training.
        We compute them DIRECTLY from graph structure.
        """
        pagerank = self.builder.compute_pagerank()
        communities = self.builder.detect_communities()
        
        # Max values for normalization
        max_pr = max(pagerank.values()) if pagerank else 1e-8
        max_freq = max(self.builder.word_freq.values()) if self.builder.word_freq else 1
        degrees = {}
        for word in self.vocab:
            if word in self.builder.graph:
                degrees[word] = self.builder.graph.degree(word)
            else:
                degrees[word] = 0
        max_degree = max(degrees.values()) if degrees else 1
        
        # R: Receptance — sigmoid(pagerank) — how much to accept
        self.R = torch.zeros(self.vocab_size, self.dim)
        for word, idx in self.vocab.items():
            pr = pagerank.get(word, 0) / (max_pr + 1e-8)
            self.R[idx] = torch.sigmoid(torch.full((self.dim,), pr * 3.0 - 1.5))
        
        # W: Decay — exp(-1 / (degree + 1)) — how fast to forget
        self.W = torch.zeros(self.vocab_size, self.dim)
        for word, idx in self.vocab.items():
            deg = degrees.get(word, 0)
            decay = math.exp(-1.0 / (deg + 1))
            self.W[idx] = torch.full((self.dim,), decay)
        
        # K: Key — graph embedding
        self.K = self.graph_emb.embeddings.clone()
        
        # V: Value — neighborhood mean embedding
        self.V = torch.zeros(self.vocab_size, self.dim)
        for word, idx in self.vocab.items():
            if word in self.builder.graph:
                neighbors = [n for n in self.builder.graph.successors(word)
                            if self.builder.graph.nodes.get(n, {}).get("type") == "WORD"
                            and n in self.vocab]
                if neighbors:
                    neighbor_embs = torch.stack([
                        self.graph_emb.embeddings[self.vocab[n]] for n in neighbors
                    ])
                    self.V[idx] = neighbor_embs.mean(dim=0)
                else:
                    self.V[idx] = self.K[idx]  # Self-loop fallback
            else:
                self.V[idx] = self.K[idx]
        
        # A: Learning Rate — cluster similarity
        self.A = torch.zeros(self.vocab_size, self.dim)
        for word, idx in self.vocab.items():
            comm = communities.get(word, -1)
            # Words in larger communities have higher learning rate
            comm_size = sum(1 for w, c in communities.items() if c == comm)
            a_val = torch.sigmoid(torch.full((self.dim,), comm_size / (self.vocab_size + 1e-8) * 5.0))
            self.A[idx] = a_val
        
        # G: Gate — co-occurrence based
        self.G = torch.zeros(self.vocab_size, self.dim)
        for word, idx in self.vocab.items():
            if word in self.builder.cooccurrence:
                total_cooccur = sum(self.builder.cooccurrence[word].values())
                g_val = torch.sigmoid(torch.full((self.dim,), math.log1p(total_cooccur) / 5.0 - 1.0))
            else:
                g_val = torch.full((self.dim,), 0.3)  # Low default gate
            self.G[idx] = g_val
    
    def reset_state(self):
        """Reset the WKV state matrix."""
        self.state = torch.zeros(self.dim, self.dim)
    
    def step(self, word_idx: int) -> torch.Tensor:
        """
        Single RWKV-7 WKV step using graph-derived values.
        
        Returns: output vector [dim]
        """
        if word_idx < 0 or word_idx >= self.vocab_size:
            return torch.zeros(self.dim)
        
        r = self.R[word_idx].unsqueeze(-1)   # [dim, 1]
        w = self.W[word_idx].unsqueeze(-1)   # [dim, 1]
        k = self.K[word_idx].unsqueeze(0)    # [1, dim]
        v = self.V[word_idx].unsqueeze(-1)   # [dim, 1]
        a = self.A[word_idx].unsqueeze(-1)   # [dim, 1]
        g = self.G[word_idx].unsqueeze(-1)   # [dim, 1]
        
        # Generalized Delta Rule (RWKV-7 Goose)
        # S = S * diag(w) - (S @ (g * a)) @ g^T + v @ k
        self.state = self.state * w.T                              # Decay
        self.state = self.state - (self.state @ (g * a)) @ g.T    # Selective removal
        self.state = self.state + (v @ k) * 0.1                   # New info (scaled down)
        
        # STATE NORMALIZATION — prevents explosion over long sequences
        # Without this, state norm goes 0.6 → 33 million in 7 steps
        state_norm = self.state.norm()
        if state_norm > 10.0:
            self.state = self.state * (10.0 / state_norm)
        
        # Clamp extreme values
        self.state = torch.clamp(self.state, -5.0, 5.0)
        
        # Output
        output = (self.state @ r).squeeze(-1)  # [dim]
        
        return output
    
    def compute_logits(self, output: torch.Tensor) -> torch.Tensor:
        """
        Convert WKV output to word logits via similarity with all embeddings.
        logits[i] = cosine_sim(output, embedding[i])
        """
        # Normalize
        out_norm = output / (output.norm() + 1e-8)
        emb_norm = self.graph_emb.embeddings / (self.graph_emb.embeddings.norm(dim=1, keepdim=True) + 1e-8)
        
        # Cosine similarity as logits
        logits = emb_norm @ out_norm  # [vocab_size]
        
        # Scale up (cosine sim is in [-1, 1], we need sharper distribution)
        logits = logits * 10.0
        
        return logits


class TextGenerator:
    """
    Full text generation pipeline:
      Word Graph → Graph-RWKV-7 WKV → Token Prediction → Grammar Filter → Output
    """
    
    def __init__(self, graph_builder: WordGraphBuilder, dim: int = 128):
        self.builder = graph_builder
        self.rwkv = GraphRWKV7(graph_builder, dim)
        self.grammar = graph_builder.get_grammar_model()
        self.vocab = graph_builder.vocab
        self.id_to_word = graph_builder.id_to_word
        self.vocab_size = len(self.vocab)
    
    def generate(self, 
                 prompt: str = "", 
                 max_tokens: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 20,
                 top_p: float = 0.9,
                 grammar_weight: float = 0.3,
                 repetition_penalty: float = 1.2,
                 seed: Optional[int] = None) -> str:
        """
        Generate text using graph-derived RWKV-7 WKV values.
        
        Args:
            prompt: Starting text (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            grammar_weight: Weight for grammar-based logit adjustment
            repetition_penalty: Penalty for repeating recent tokens
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.rwkv.reset_state()
        
        # Tokenize prompt
        tokenizer = self.builder.tokenizer
        if prompt:
            prompt_tokens = tokenizer.tokenize(prompt)
        else:
            # Start with a random high-PageRank word
            god_nodes = self.builder.get_god_nodes(5)
            if god_nodes:
                prompt_tokens = [god_nodes[np.random.randint(len(god_nodes))][0]]
            else:
                prompt_tokens = [list(self.vocab.keys())[0]]
        
        # Process prompt through WKV
        generated = list(prompt_tokens)
        last_pos = "NOUN"
        recent_tokens = []
        
        for token in prompt_tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                self.rwkv.step(idx)
                last_pos = self.builder.word_to_pos.get(token, "NOUN")
                recent_tokens.append(idx)
        
        # Get community info for topic continuity
        communities = self.builder.detect_communities()
        
        # Generate new tokens
        for step in range(max_tokens):
            # Get current token
            current_word = generated[-1] if generated else ""
            current_idx = self.vocab.get(current_word, 0)
            
            # WKV step
            output = self.rwkv.step(current_idx)
            
            # Compute logits from WKV
            logits = self.rwkv.compute_logits(output)
            
            # === BIGRAM BOOST (strongest signal) ===
            # The graph already stores observed bigrams — use them heavily
            if current_word in self.builder.graph:
                for successor in self.builder.graph.successors(current_word):
                    edge_data = self.builder.graph[current_word][successor]
                    if edge_data.get("type") == "FOLLOWS" and successor in self.vocab:
                        # Strong boost for observed bigrams
                        bigram_score = math.log1p(edge_data.get("freq", 1)) * 2.0
                        logits[self.vocab[successor]] += bigram_score
            
            # === TRIGRAM BOOST ===
            if len(generated) >= 2:
                prev_word = generated[-2]
                for (w1, w2, w3), freq in self.builder.trigram_freq.items():
                    if w1 == prev_word and w2 == current_word and w3 in self.vocab:
                        logits[self.vocab[w3]] += math.log1p(freq) * 3.0
            
            # === GRAMMAR FILTER ===
            if grammar_weight > 0 and last_pos in self.grammar:
                grammar_bonus = torch.zeros(self.vocab_size)
                valid_next_pos = self.grammar[last_pos]
                
                for word, word_idx in self.vocab.items():
                    word_pos = self.builder.word_to_pos.get(word, "NOUN")
                    if word_pos in valid_next_pos:
                        grammar_bonus[word_idx] = valid_next_pos[word_pos] * grammar_weight * 5.0
                    else:
                        # Penalty for grammatically invalid transitions
                        grammar_bonus[word_idx] = -grammar_weight * 2.0
                
                logits = logits + grammar_bonus
            
            # === CO-OCCURRENCE BOOST ===
            # Boost words that co-occur with recent context (not just current word)
            context_words = generated[-5:] if len(generated) >= 5 else generated
            for ctx_word in context_words:
                if ctx_word in self.builder.cooccurrence:
                    for coword, freq in self.builder.cooccurrence[ctx_word].items():
                        if coword in self.vocab:
                            logits[self.vocab[coword]] += math.log1p(freq) * 0.3
            
            # === TOPIC CONTINUITY ===
            # Boost words in the same Leiden cluster as recent context
            if communities:
                recent_comms = [communities.get(w, -1) for w in generated[-3:] if w in communities]
                if recent_comms:
                    dominant_comm = max(set(recent_comms), key=recent_comms.count)
                    for word, word_idx in self.vocab.items():
                        if communities.get(word, -2) == dominant_comm:
                            logits[word_idx] += 0.5  # Mild topic continuity boost
            
            # === REPETITION PENALTY ===
            for recent_idx in recent_tokens[-10:]:
                logits[recent_idx] /= repetition_penalty
            # Extra penalty for immediate repetition
            if recent_tokens:
                logits[recent_tokens[-1]] /= (repetition_penalty * 2.0)
            
            # === SAMPLING ===
            next_idx = self._sample(logits, temperature, top_k, top_p)
            next_word = self.id_to_word.get(next_idx, "<unk>")
            
            # Stop conditions
            if next_word in {".", "!", "?"} and len(generated) > len(prompt_tokens) + 5:
                generated.append(next_word)
                break
            
            generated.append(next_word)
            last_pos = self.builder.word_to_pos.get(next_word, "NOUN")
            recent_tokens.append(next_idx)
        
        return " ".join(generated)
    
    def _sample(self, logits: torch.Tensor, temperature: float, 
                top_k: int, top_p: float) -> int:
        """Top-k + Nucleus sampling."""
        # Temperature
        logits = logits / max(temperature, 1e-8)
        
        # Top-k filtering
        if top_k > 0 and top_k < len(logits):
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[-1]
            logits = torch.where(logits >= min_top_k, logits, torch.full_like(logits, -1e9))
        
        # Softmax
        probs = F.softmax(logits, dim=-1)
        
        # Nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff
            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            # Sample from filtered distribution
            idx = torch.multinomial(sorted_probs, 1).item()
            return sorted_indices[idx].item()
        
        # Standard sampling
        return torch.multinomial(probs, 1).item()


# =============================================================================
# CONVENIENCE API
# =============================================================================

def create_generator(texts: List[str], dim: int = 128, 
                     window_size: int = 5, min_freq: int = 1) -> TextGenerator:
    """
    One-shot: Build word graph + Create text generator.
    
    Usage:
        gen = create_generator(["your corpus text here..."])
        print(gen.generate("hello", max_tokens=30))
    """
    from word_graph import build_word_graph
    
    builder = build_word_graph(texts, window_size=window_size, min_freq=min_freq)
    return TextGenerator(builder, dim=dim)
