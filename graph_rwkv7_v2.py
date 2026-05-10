"""
Graph-RWKV-7 V2: Pure Graph Weight Matrices — Zero Template Matching
=====================================================================
Graph structure IS the weight matrices. No bigram boost, no grammar filter,
no POS rules, no heuristics.

RWKV-7 Generalized Delta Rule with graph-derived weight matrices:
  r = sigmoid(W_r @ x_mixed)    — Context-dependent receptance
  w = exp(clamp(W_w @ x_mixed)) — Context-dependent decay
  k = W_k @ x_mixed             — Context-dependent key
  v = W_v @ x_mixed             — Context-dependent value
  a = sigmoid(W_a @ x_mixed)    — Context-dependent learning rate
  g = sigmoid(W_g @ x_mixed)    — Context-dependent gate

State update:
  S = S * diag(w) - (S @ (g * a)) @ g^T + v^T @ k
  output = S @ r

ALL weight matrices W_r...W_g are [dim, dim], constructed from graph.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from token_graph import TokenGraph


class GraphRWKV7V2:
    """
    RWKV-7 with graph-derived weight matrices.
    
    Key difference from V1:
      V1: R[word_idx] = fixed scalar per word (template matching)
      V2: r = sigmoid(W_r @ x) where W_r is from graph (context-dependent)
    
    Infinite context: state never resets within a session.
    """
    
    def __init__(self, token_graph: TokenGraph):
        self.graph = token_graph
        self.dim = token_graph.dim
        self.vocab_size = token_graph.vocab_size
        
        # Convert graph matrices to torch tensors
        self.E = torch.tensor(token_graph.embeddings, dtype=torch.float32)  # [V, dim]
        self.W_r = torch.tensor(token_graph.W_r, dtype=torch.float32)      # [dim, dim]
        self.W_w = torch.tensor(token_graph.W_w, dtype=torch.float32)
        self.W_k = torch.tensor(token_graph.W_k, dtype=torch.float32)
        self.W_v = torch.tensor(token_graph.W_v, dtype=torch.float32)
        self.W_a = torch.tensor(token_graph.W_a, dtype=torch.float32)
        self.W_g = torch.tensor(token_graph.W_g, dtype=torch.float32)
        
        # Time-mixing ratio (normally learned, we derive from graph density)
        edge_density = len(token_graph.bigram_freq) / (self.vocab_size ** 2 + 1e-8)
        self.mix_ratio = min(0.8, max(0.3, edge_density * 100))
        
        # Frequency log-prior (same as output bias in trained LMs)
        # NOT a rule — it's the unigram distribution FROM THE DATA
        total = sum(token_graph.token_freq.values()) or 1
        freq_prior = torch.zeros(self.vocab_size)
        for token, idx in token_graph.vocab.items():
            p = token_graph.token_freq[token] / total
            freq_prior[idx] = math.log(p + 1e-10)
        # Normalize to [-1, 1] range
        freq_prior = freq_prior - freq_prior.mean()
        freq_prior = freq_prior / (freq_prior.abs().max() + 1e-8)
        self.freq_prior = freq_prior
        
        # RWKV-7 state — [dim, dim] — INFINITE CONTEXT (never reset)
        self.state = torch.zeros(self.dim, self.dim)
        
        # Previous token embedding for time-mixing
        self.x_prev = torch.zeros(self.dim)
        
        # Step counter for state management
        self.step_count = 0
    
    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Get spectral embedding for a token."""
        if 0 <= token_id < self.vocab_size:
            return self.E[token_id]
        return torch.zeros(self.dim)
    
    def step(self, token_id: int) -> torch.Tensor:
        """
        Single RWKV-7 step with GRAPH-DERIVED weight matrices.
        
        This is NOT a lookup. W @ x gives different results for same token
        in different contexts because x_mixed depends on x_prev.
        
        Args:
            token_id: Index of current token
            
        Returns:
            output: [dim] output vector
        """
        x = self.get_embedding(token_id)
        
        # Time-mixing: blend current with previous (captures sequential context)
        x_mixed = x * self.mix_ratio + self.x_prev * (1 - self.mix_ratio)
        
        # CONTEXT-DEPENDENT projections (graph weights × mixed input)
        r = torch.sigmoid(self.W_r @ x_mixed)                          # [dim]
        w = torch.exp(torch.clamp(self.W_w @ x_mixed, -3.0, 0.0))     # [dim] decay in (0, 1)
        k = self.W_k @ x_mixed                                         # [dim]
        v = self.W_v @ x_mixed                                         # [dim]
        a = torch.sigmoid(self.W_a @ x_mixed)                          # [dim]
        g = torch.sigmoid(self.W_g @ x_mixed)                          # [dim]
        
        # Generalized Delta Rule (RWKV-7 Goose)
        # S = diag(w) * S - (S @ (g * a)) @ g^T + v^T @ k
        r_col = r.unsqueeze(-1)    # [dim, 1]
        k_row = k.unsqueeze(0)     # [1, dim]
        v_col = v.unsqueeze(-1)    # [dim, 1]
        a_col = a.unsqueeze(-1)    # [dim, 1]
        g_col = g.unsqueeze(-1)    # [dim, 1]
        
        # State update
        self.state = self.state * w.unsqueeze(0)                        # Decay
        self.state = self.state - (self.state @ (g_col * a_col)) @ g_col.T  # Selective removal
        self.state = self.state + (v_col @ k_row) * 0.15               # New information
        
        # State normalization (prevents explosion, enables infinite context)
        state_norm = self.state.norm()
        if state_norm > 5.0:
            self.state = self.state * (5.0 / state_norm)
        
        # Output
        output = (self.state @ r_col).squeeze(-1)  # [dim]
        
        # Update previous embedding for next step's time-mixing
        self.x_prev = x.clone()
        self.step_count += 1
        
        return output
    
    def compute_logits(self, output: torch.Tensor) -> torch.Tensor:
        """
        Convert WKV output to next-token logits.
        
        Blends:
          - WKV cosine similarity (context-dependent signal)
          - Frequency log-prior (data distribution, like output bias in trained LMs)
        
        The blend ratio shifts toward WKV as more context accumulates.
        """
        out_norm = output / (output.norm() + 1e-8)
        emb_norm = self.E / (self.E.norm(dim=1, keepdim=True) + 1e-8)
        
        # WKV-based logits (context-dependent)
        wkv_logits = emb_norm @ out_norm  # [vocab_size]
        wkv_logits = wkv_logits * 10.0
        
        # Blend with frequency prior
        # Early: balanced. Later: WKV dominates as context accumulates
        wkv_weight = min(0.85, 0.5 + self.step_count * 0.01)
        prior_weight = 1.0 - wkv_weight
        
        logits = wkv_logits * wkv_weight + self.freq_prior * prior_weight * 8.0
        
        return logits
    
    def soft_reset(self, decay: float = 0.1):
        """
        Soft reset — decay state instead of clearing it.
        Use between conversations but within a session.
        """
        self.state = self.state * decay
        self.x_prev = self.x_prev * decay


class InfiniteContextGenerator:
    """
    Text generator with infinite context session.
    
    Session state persists across all generate() calls.
    The model REMEMBERS everything said in the session.
    
    No template matching. No hardcoded rules.
    Pure WKV state → embedding similarity → sampling.
    """
    
    def __init__(self, token_graph: TokenGraph):
        self.graph = token_graph
        self.rwkv = GraphRWKV7V2(token_graph)
        self.vocab = token_graph.vocab
        self.id_to_token = token_graph.id_to_token
        self.vocab_size = token_graph.vocab_size
        
        # Session history
        self.session_tokens = []
    
    def generate(self, 
                 prompt: str = "",
                 max_tokens: int = 60,
                 temperature: float = 0.7,
                 top_k: int = 20,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.3,
                 seed: Optional[int] = None) -> str:
        """
        Generate text using infinite-context graph-RWKV-7.
        
        NO bigram boost.
        NO grammar filter.
        NO template matching.
        ONLY WKV state → cosine similarity → sampling.
        
        State carries over from previous generate() calls (infinite context).
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Tokenize prompt
        if prompt:
            prompt_tokens = TokenGraph.tokenize(prompt)
        else:
            # Start with highest PageRank content token
            top = self.graph.top_tokens(20)
            content_tokens = [t for t, s in top if len(t) > 2]
            if content_tokens:
                prompt_tokens = [content_tokens[0]]
            else:
                prompt_tokens = [top[0][0]] if top else ["the"]
        
        # Feed prompt through WKV (state updates carry context)
        for token in prompt_tokens:
            if token in self.vocab:
                self.rwkv.step(self.vocab[token])
                self.session_tokens.append(token)
        
        generated = list(prompt_tokens)
        
        # Generate new tokens
        for _ in range(max_tokens):
            current_token = generated[-1] if generated else ""
            current_id = self.vocab.get(current_token, 0)
            
            # WKV step — state carries ALL context
            output = self.rwkv.step(current_id)
            
            # Pure logits from WKV output (NO boosts)
            logits = self.rwkv.compute_logits(output)
            
            # Repetition penalty on recent tokens
            recent = self.session_tokens[-20:] + generated[-20:]
            seen_ids = set()
            for t in recent:
                if t in self.vocab:
                    tid = self.vocab[t]
                    count = recent.count(t)
                    penalty = repetition_penalty ** min(count, 3)
                    logits[tid] = logits[tid] / penalty
                    seen_ids.add(tid)
            
            # Sample
            next_id = self._sample(logits, temperature, top_k, top_p)
            next_token = self.id_to_token.get(next_id, "")
            
            if not next_token:
                continue
            
            generated.append(next_token)
            self.session_tokens.append(next_token)
            
            # Stop at sentence end (but not too early)
            if next_token in {'.', '!', '?'} and len(generated) > len(prompt_tokens) + 8:
                break
        
        return self._detokenize(generated)
    
    def _sample(self, logits: torch.Tensor, temperature: float,
                top_k: int, top_p: float) -> int:
        """Top-k + nucleus sampling."""
        logits = logits / max(temperature, 1e-8)
        
        # Top-k
        if 0 < top_k < len(logits):
            top_vals, _ = torch.topk(logits, top_k)
            threshold = top_vals[-1]
            logits = torch.where(logits >= threshold, logits, torch.full_like(logits, -1e9))
        
        probs = F.softmax(logits, dim=-1)
        
        # Nucleus (top-p)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumulative - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-8)
            chosen = torch.multinomial(sorted_probs, 1).item()
            return sorted_idx[chosen].item()
        
        return torch.multinomial(probs, 1).item()
    
    @staticmethod
    def _detokenize(tokens: List[str]) -> str:
        """
        Smart detokenization — join tokens with spaces,
        but attach punctuation to previous word.
        """
        if not tokens:
            return ""
        
        result = [tokens[0]]
        punct = set('.,;:!?)]}\'"')
        open_punct = set('([{')
        
        for t in tokens[1:]:
            if t in punct:
                result.append(t)  # No space before closing punct
            elif result and result[-1] in open_punct:
                result.append(t)  # No space after opening punct
            else:
                result.append(' ')
                result.append(t)
        
        return ''.join(result)
    
    def session_stats(self) -> dict:
        """Stats about the current session."""
        return {
            'session_tokens': len(self.session_tokens),
            'state_norm': self.rwkv.state.norm().item(),
            'step_count': self.rwkv.step_count,
            'unique_tokens_seen': len(set(self.session_tokens)),
        }


# =============================================================================
# CONVENIENCE API
# =============================================================================

def build_generator(texts: List[str], dim: int = 128, 
                    min_freq: int = 2, window_size: int = 5) -> InfiniteContextGenerator:
    """
    One-shot: Build token graph + Create infinite-context generator.
    
    Works on any data:
        gen = build_generator(["English text...", "Python code...", "Hindi text..."])
        print(gen.generate("hello"))
    """
    graph = TokenGraph(dim=dim, window_size=window_size)
    graph.add_texts(texts)
    graph.build(min_freq=min_freq)
    return InfiniteContextGenerator(graph)
