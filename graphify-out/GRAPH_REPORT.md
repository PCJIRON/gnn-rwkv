# Graph Report - gnn  (2026-05-05)

## Corpus Check
- 23 files · ~13,818 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 264 nodes · 401 edges · 44 communities (22 shown, 22 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `032247c0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]

## God Nodes (most connected - your core abstractions)
1. `SocialNarrativeModel` - 20 edges
2. `SemanticMemoryGraph` - 18 edges
3. `train_brain()` - 16 edges
4. `Persistent GNN Brain Chat Interface` - 14 edges
5. `PersistentGNNBrain` - 13 edges
6. `get_data_from_text()` - 12 edges
7. `AutoTrainer` - 11 edges
8. `start_chat()` - 9 edges
9. `tokenize()` - 9 edges
10. `EfficientTrainer` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_model_uses_active_node_mask_for_lif_routing()` --calls--> `SocialNarrativeModel`  [INFERRED]
  tests/test_semantic_routing.py → src/gnn_rwkv_story_gen.py
- `AutoTrainer` --uses--> `SocialNarrativeModel`  [INFERRED]
  scripts/auto_train_loop.py → src/gnn_rwkv_story_gen.py
- `train_on_batch()` --calls--> `get_data_from_text()`  [INFERRED]
  scripts/batch_train_10.py → src/gnn_rwkv_story_gen.py
- `main()` --calls--> `SocialNarrativeModel`  [INFERRED]
  scripts/batch_train_10.py → src/gnn_rwkv_story_gen.py
- `main()` --calls--> `PersistentGNNBrain`  [INFERRED]
  scripts/batch_train_10.py → src/chat.py

## Communities (44 total, 22 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (32): Active Brain Path (active_brain.json), Adaptive Node Router (TLT), Adaptive Social Brain Architecture, Auto Training Loop with Web Crawl, CPU Efficient Trainer, Dynamic GNN RWKV Block with LIF Neurons, SALT-based Efficient Trainer, Meta Path (brain_meta.json) (+24 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (13): build_memory_from_text(), cluster_mask(), compact_phrase(), from_dict(), load(), node_id(), normalize_label(), SemanticMemoryGraph (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.14
Nodes (15): AdaptiveSocialBrain, basic_words(), build_hybrid_vocab(), char_from_token(), char_token(), decode_tokens(), DynamicGNNRWKVBlock, encode_text() (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (12): AdaptiveNodeRouter, EfficientTrainer, Efficient Training Module - SALT + TLT methodology Based on: 1. SALT: Small mode, Select hard examples using SALT methodology:         - Compute per-token losses, Two-stage training: KD first, then standard, TLT-inspired adaptive node routing:     - Uses LIF neurons efficiently     - Foc, Track which nodes are most important, Adaptive node activation based on input difficulty:         - Hard inputs: activ (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.22
Nodes (11): ChatMessage, clean_crawled_text(), crawl_and_train(), get_search_results(), is_saved_brain_ready(), load_active_brain_paths(), PersistentGNNBrain, render_chat() (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (7): AutoTrainer, Auto Training Loop with Web Crawl + Evaluate after each batch - Crawls web data, Evaluate relevance and grammar scores, Crawl web for topic using crawl4ai or fallback, Train for specified epochs, Save model, vocab, meta, Main training loop with evaluation after each batch

### Community 6 - "Community 6"
Cohesion: 0.23
Nodes (13): _backup_existing_brain(), _build_base_language_corpus(), incremental_train(), _load_active_paths(), _read_limited_training_text(), _save_active_manifest(), _save_weights_versioned(), _tokenize_training_text() (+5 more)

### Community 7 - "Community 7"
Cohesion: 0.2
Nodes (14): clean_text(), crawl_urls(), evaluate_model(), fallback_fetch(), get_search_results(), main(), Web Crawl + Train + Evaluate Loop - Fetch data using crawl4ai in batches - Train, Train model on a batch of text (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (9): SocialNarrativeModel, CPUEfficientTrainer, CPU-Friendly Efficient Training Module Optimizations that work on CPU only:  1., Two-stage training on CPU, CPU-efficient training - no GPU dependencies., CPU-optimized efficient training using SALT methodology.     No GPU-specific cod, KD loss with temperature scaling, Evaluate which examples are hard/easy.         Returns indices sorted by difficu (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (8): Test if NLP model produces the expected embedding dimension (384 for MiniLM)., Test if the GNN model produces a single output per node., Test if we can dynamically add a node and its edges., Test if a single training step reduces loss (basic sanity check)., test_dynamic_node_addition(), test_gnn_model_forward(), test_model_training_step(), test_nlp_embedding_shape()

### Community 10 - "Community 10"
Cohesion: 0.44
Nodes (8): chunk_text(), crawl_url(), evaluate_chat(), grammar_score(), main(), relevance_score(), resolve_seed(), vocab_size_mb()

### Community 11 - "Community 11"
Cohesion: 0.46
Nodes (7): clean_text(), crawl_urls(), fallback_fetch(), get_search_results(), main(), Batch Training + Chat Evaluation - 10 batches with 2 epochs each - Chat evaluati, train_on_batch()

### Community 12 - "Community 12"
Cohesion: 0.46
Nodes (7): clean_text(), crawl_urls(), fallback_fetch(), get_search_results(), main(), Fast Batch Training + Chat Evaluation - 10 batches with 2 epochs each - 20 URLs, train_on_batch()

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (3): Semantic Memory Knowledge Graph, Build Memory from Text, Cluster Mask Generator

## Knowledge Gaps
- **71 isolated node(s):** `Auto Training Loop with Web Crawl + Evaluate after each batch - Crawls web data`, `Evaluate relevance and grammar scores`, `Crawl web for topic using crawl4ai or fallback`, `Train for specified epochs`, `Save model, vocab, meta` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SocialNarrativeModel` connect `Community 8` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 11`, `Community 12`?**
  _High betweenness centrality (0.334) - this node is a cross-community bridge._
- **Why does `test_model_uses_active_node_mask_for_lif_routing()` connect `Community 1` to `Community 8`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `SocialNarrativeModel` (e.g. with `AutoTrainer` and `ChatMessage`) actually correct?**
  _`SocialNarrativeModel` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `train_brain()` (e.g. with `crawl_and_train()` and `train_startup_brain()`) actually correct?**
  _`train_brain()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PersistentGNNBrain` (e.g. with `SocialNarrativeModel` and `main()`) actually correct?**
  _`PersistentGNNBrain` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Auto Training Loop with Web Crawl + Evaluate after each batch - Crawls web data`, `Evaluate relevance and grammar scores`, `Crawl web for topic using crawl4ai or fallback` to the rest of the system?**
  _71 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._