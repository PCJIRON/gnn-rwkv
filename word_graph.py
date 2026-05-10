"""
Graphify-NLP: Word Graph Builder
================================
Adapted from Graphify's code-graph approach (Karpathy-inspired)
to build HUMAN LANGUAGE graphs instead of programming language ASTs.

Pipeline:
  Text → Tokenize → Word Nodes → Edge Extraction → Graph Analysis
  
Replaces:
  - Tree-sitter AST → Regex tokenizer + POS heuristics
  - Function/Class nodes → Word/Phrase nodes  
  - Import/Call edges → Syntactic/Semantic/Co-occurrence edges
  - Code modules → Grammar clusters (Leiden)
  - Code importance → Word PageRank
"""

import re
import math
import json
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
import numpy as np


# =============================================================================
# 1. TOKENIZER (replaces Tree-sitter)
# =============================================================================

class LanguageTokenizer:
    """
    Simple but effective tokenizer for human language.
    Splits text into words, handles punctuation, preserves sentence boundaries.
    """
    
    # Basic POS heuristics (no external library needed)
    DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", 
                   "his", "her", "its", "our", "their", "some", "any", "no", "every",
                   "ye", "yeh", "wo", "is", "us", "ek", "do", "teen"}
    
    PREPOSITIONS = {"in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
                    "into", "through", "during", "before", "after", "above", "below",
                    "between", "under", "over", "mein", "par", "se", "ke", "ki", "ka",
                    "ko", "pe", "tak", "dwara"}
    
    CONJUNCTIONS = {"and", "or", "but", "nor", "yet", "so", "for", "because", "although",
                    "while", "if", "when", "aur", "ya", "lekin", "magar", "kyunki", "agar"}
    
    PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
                "them", "who", "whom", "which", "what", "that", "main", "tu", "tum",
                "aap", "vo", "hum", "kya", "kaun", "kahan", "kaise"}
    
    COMMON_VERBS = {"is", "are", "was", "were", "be", "been", "being", "have", "has", 
                    "had", "do", "does", "did", "will", "would", "could", "should",
                    "may", "might", "shall", "can", "go", "come", "make", "take",
                    "get", "know", "think", "say", "see", "give", "find", "tell",
                    "hai", "hain", "tha", "thi", "the", "ho", "karo", "karta",
                    "karti", "karte", "kar", "bolo", "dekho", "jao", "aao"}
    
    COMMON_NOUNS_SUFFIXES = ("tion", "ment", "ness", "ity", "ism", "ist", "er", "or",
                            "ence", "ance", "dom", "ship", "hood")
    
    ADJECTIVE_SUFFIXES = ("ful", "less", "ous", "ive", "able", "ible", "al", "ial",
                          "ic", "ical", "ish", "like", "ly")
    
    ADVERB_SUFFIXES = ("ly",)
    
    VERB_SUFFIXES = ("ing", "ed", "ize", "ise", "ify", "ate")
    
    def __init__(self):
        self.sentence_end_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r"[\w']+|[^\w\s]")
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into word tokens."""
        return self.word_pattern.findall(text.lower())
    
    def tokenize_sentences(self, text: str) -> List[List[str]]:
        """Split text into sentences, then tokenize each."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [self.tokenize(s) for s in sentences if s.strip()]
    
    def guess_pos(self, word: str) -> str:
        """Heuristic POS tagging without external libraries."""
        w = word.lower()
        
        if w in self.DETERMINERS:
            return "DET"
        if w in self.PREPOSITIONS:
            return "PREP"
        if w in self.CONJUNCTIONS:
            return "CONJ"
        if w in self.PRONOUNS:
            return "PRON"
        if w in self.COMMON_VERBS:
            return "VERB"
        
        # Suffix-based heuristics
        if any(w.endswith(s) for s in self.VERB_SUFFIXES):
            return "VERB"
        if any(w.endswith(s) for s in self.ADVERB_SUFFIXES) and len(w) > 4:
            return "ADV"
        if any(w.endswith(s) for s in self.ADJECTIVE_SUFFIXES):
            return "ADJ"
        if any(w.endswith(s) for s in self.COMMON_NOUNS_SUFFIXES):
            return "NOUN"
        
        # Punctuation
        if re.match(r'^[^\w\s]+$', w):
            return "PUNCT"
        
        # Default: NOUN (most common open class)
        return "NOUN"


# =============================================================================
# 2. WORD GRAPH BUILDER (replaces Graphify's AST Graph)
# =============================================================================

class WordGraphBuilder:
    """
    Builds a knowledge graph from text, Graphify-style.
    
    Instead of code AST → graph, we do:
      Text → Word nodes + POS nodes + Syntactic edges + Semantic edges
    """
    
    def __init__(self, window_size: int = 5, min_freq: int = 1):
        self.window_size = window_size
        self.min_freq = min_freq
        self.tokenizer = LanguageTokenizer()
        self.graph = nx.DiGraph()
        
        # Statistics
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.trigram_freq = Counter()
        self.cooccurrence = defaultdict(Counter)
        self.pos_transitions = defaultdict(Counter)
        
        # Vocab
        self.vocab = {}  # word → id
        self.id_to_word = {}  # id → word
        self.word_to_pos = {}  # word → POS tag
        
        # Cache
        self._file_hashes = {}  # SHA256 caching like Graphify
    
    def _hash_text(self, text: str) -> str:
        """SHA256 hash for incremental updates (Graphify-style caching)."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def add_text(self, text: str, source_id: str = "corpus") -> bool:
        """
        Add text to the graph. Returns True if new content was processed.
        Uses SHA256 caching like Graphify to avoid re-processing.
        """
        text_hash = self._hash_text(text)
        if source_id in self._file_hashes and self._file_hashes[source_id] == text_hash:
            return False  # Already processed
        
        self._file_hashes[source_id] = text_hash
        
        # Step 1: Tokenize
        sentences = self.tokenizer.tokenize_sentences(text)
        
        # Step 2: Count frequencies
        for sentence in sentences:
            for word in sentence:
                self.word_freq[word] += 1
            
            # Bigrams
            for i in range(len(sentence) - 1):
                self.bigram_freq[(sentence[i], sentence[i+1])] += 1
            
            # Trigrams
            for i in range(len(sentence) - 2):
                self.trigram_freq[(sentence[i], sentence[i+1], sentence[i+2])] += 1
            
            # Co-occurrence within window
            for i, word in enumerate(sentence):
                start = max(0, i - self.window_size)
                end = min(len(sentence), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        self.cooccurrence[word][sentence[j]] += 1
            
            # POS transitions
            pos_tags = [self.tokenizer.guess_pos(w) for w in sentence]
            for i in range(len(pos_tags) - 1):
                self.pos_transitions[pos_tags[i]][pos_tags[i+1]] += 1
        
        # Step 3: Build vocab
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.id_to_word[idx] = word
                self.word_to_pos[word] = self.tokenizer.guess_pos(word)
        
        return True
    
    def build_graph(self):
        """
        Construct the full word graph with Graphify-style analysis.
        
        Node types: WORD, POS_TAG
        Edge types: FOLLOWS, CO_OCCURS, HAS_POS, POS_TRANSITION
        """
        self.graph.clear()
        
        # --- WORD NODES ---
        for word, idx in self.vocab.items():
            freq = self.word_freq[word]
            pos = self.word_to_pos.get(word, "NOUN")
            self.graph.add_node(
                word,
                type="WORD",
                idx=idx,
                freq=freq,
                log_freq=math.log1p(freq),
                pos=pos,
                confidence="EXTRACTED"  # Graphify-style confidence tag
            )
        
        # --- POS TAG NODES ---
        all_pos = set(self.word_to_pos.values())
        for pos in all_pos:
            self.graph.add_node(
                f"POS:{pos}",
                type="POS_TAG",
                tag=pos,
                confidence="EXTRACTED"
            )
        
        # --- SEQUENTIAL EDGES (FOLLOWS) ---
        # Bigram edges: word_a → word_b
        for (w1, w2), freq in self.bigram_freq.items():
            if w1 in self.vocab and w2 in self.vocab:
                weight = math.log1p(freq)
                self.graph.add_edge(
                    w1, w2,
                    type="FOLLOWS",
                    weight=weight,
                    freq=freq,
                    confidence="EXTRACTED"
                )
        
        # --- CO-OCCURRENCE EDGES ---
        for w1, neighbors in self.cooccurrence.items():
            if w1 not in self.vocab:
                continue
            for w2, freq in neighbors.most_common(20):  # Top 20 co-occurrences
                if w2 in self.vocab and w1 != w2 and freq >= 2:
                    # Only add if not already a FOLLOWS edge (avoid duplicates)
                    if not self.graph.has_edge(w1, w2) or self.graph[w1][w2].get("type") != "FOLLOWS":
                        pmi = self._compute_pmi(w1, w2, freq)
                        if pmi > 0:  # Only positive association
                            self.graph.add_edge(
                                w1, w2,
                                type="CO_OCCURS",
                                weight=pmi,
                                freq=freq,
                                confidence="INFERRED"
                            )
        
        # --- POS ASSIGNMENT EDGES ---
        for word, pos in self.word_to_pos.items():
            if word in self.vocab:
                self.graph.add_edge(
                    word, f"POS:{pos}",
                    type="HAS_POS",
                    weight=1.0,
                    confidence="EXTRACTED"
                )
        
        # --- POS TRANSITION EDGES (Grammar Model) ---
        for pos1, transitions in self.pos_transitions.items():
            total = sum(transitions.values())
            for pos2, freq in transitions.items():
                prob = freq / total
                if prob > 0.01:  # Only significant transitions
                    self.graph.add_edge(
                        f"POS:{pos1}", f"POS:{pos2}",
                        type="POS_TRANSITION",
                        weight=prob,
                        freq=freq,
                        confidence="EXTRACTED"
                    )
        
        return self.graph
    
    def _compute_pmi(self, w1: str, w2: str, cooccur_freq: int) -> float:
        """Pointwise Mutual Information — measures semantic association strength."""
        total = sum(self.word_freq.values())
        if total == 0:
            return 0.0
        p_w1 = self.word_freq[w1] / total
        p_w2 = self.word_freq[w2] / total
        p_joint = cooccur_freq / total
        
        if p_w1 == 0 or p_w2 == 0 or p_joint == 0:
            return 0.0
        
        return math.log2(p_joint / (p_w1 * p_w2))
    
    # =========================================================================
    # GRAPHIFY-STYLE ANALYSIS
    # =========================================================================
    
    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """
        PageRank — identifies the most important/connected words.
        Graphify uses this to find "God Nodes" (architectural pillars).
        For language: these are the hub words everything flows through.
        """
        word_graph = self.graph.subgraph(
            [n for n, d in self.graph.nodes(data=True) if d.get("type") == "WORD"]
        )
        if len(word_graph) == 0:
            return {}
        
        try:
            pr = nx.pagerank(word_graph, alpha=alpha, weight="weight")
        except nx.PowerIterationFailedConvergence:
            pr = {n: 1.0 / len(word_graph) for n in word_graph.nodes()}
        
        # Store in graph
        for node, score in pr.items():
            self.graph.nodes[node]["pagerank"] = score
        
        return pr
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Leiden-style community detection.
        Groups words into semantic/topic communities.
        
        We use Louvain (built into NetworkX) as Leiden approximation.
        """
        word_graph = self.graph.subgraph(
            [n for n, d in self.graph.nodes(data=True) if d.get("type") == "WORD"]
        ).to_undirected()
        
        if len(word_graph) == 0:
            return {}
        
        try:
            communities = nx.community.louvain_communities(
                word_graph, weight="weight", resolution=resolution, seed=42
            )
        except Exception:
            # Fallback: each node is its own community
            communities = [{n} for n in word_graph.nodes()]
        
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
                self.graph.nodes[node]["community"] = i
        
        return community_map
    
    def get_god_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Graphify's 'God Nodes' — most connected/important words."""
        pr = self.compute_pagerank()
        sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def get_grammar_model(self) -> Dict[str, Dict[str, float]]:
        """
        Extract POS transition probabilities for grammar enforcement.
        Returns: {pos_tag: {next_pos: probability}}
        """
        grammar = {}
        for pos1, transitions in self.pos_transitions.items():
            total = sum(transitions.values())
            grammar[pos1] = {
                pos2: freq / total 
                for pos2, freq in transitions.items()
            }
        return grammar
    
    # =========================================================================
    # EXPORT (Graphify-compatible)
    # =========================================================================
    
    def export_json(self, path: str):
        """Export graph as JSON (like Graphify's graph.json)."""
        data = nx.node_link_data(self.graph)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def generate_report(self) -> str:
        """Generate GRAPH_REPORT.md (like Graphify's output)."""
        word_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "WORD"]
        
        report = []
        report.append("# Language Graph Report")
        report.append(f"\n## Statistics")
        report.append(f"- Total words in vocab: {len(self.vocab)}")
        report.append(f"- Total nodes: {self.graph.number_of_nodes()}")
        report.append(f"- Total edges: {self.graph.number_of_edges()}")
        report.append(f"- Unique bigrams: {len(self.bigram_freq)}")
        report.append(f"- Unique trigrams: {len(self.trigram_freq)}")
        
        # God Nodes
        god_nodes = self.get_god_nodes(15)
        report.append(f"\n## God Nodes (Hub Words)")
        for word, score in god_nodes:
            freq = self.word_freq[word]
            pos = self.word_to_pos.get(word, "?")
            report.append(f"- **{word}** (PageRank: {score:.4f}, freq: {freq}, POS: {pos})")
        
        # Communities
        communities = self.detect_communities()
        if communities:
            comm_groups = defaultdict(list)
            for word, comm_id in communities.items():
                comm_groups[comm_id].append(word)
            
            report.append(f"\n## Semantic Communities ({len(comm_groups)} clusters)")
            for comm_id, words in sorted(comm_groups.items()):
                top_words = sorted(words, key=lambda w: self.word_freq[w], reverse=True)[:10]
                report.append(f"- **Cluster {comm_id}**: {', '.join(top_words)}")
        
        # Grammar Model
        grammar = self.get_grammar_model()
        report.append(f"\n## Grammar Model (POS Transitions)")
        for pos, transitions in sorted(grammar.items()):
            top_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]
            trans_str = ", ".join(f"{p2}: {prob:.2f}" for p2, prob in top_trans)
            report.append(f"- {pos} → [{trans_str}]")
        
        return "\n".join(report)


# =============================================================================
# CONVENIENCE
# =============================================================================

def build_word_graph(texts: List[str], window_size: int = 5, min_freq: int = 1) -> WordGraphBuilder:
    """One-shot convenience function to build a word graph from texts."""
    builder = WordGraphBuilder(window_size=window_size, min_freq=min_freq)
    for i, text in enumerate(texts):
        builder.add_text(text, source_id=f"text_{i}")
    builder.build_graph()
    builder.compute_pagerank()
    builder.detect_communities()
    return builder
