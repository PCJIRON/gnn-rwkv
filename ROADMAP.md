# GNN-RWKV Brain Roadmap 🧠🚀

## Current Status: Phase 3 (Vectorized Apex) - COMPLETE
We have successfully implemented a high-performance, biologically-inspired hybrid architecture.

### Current Architecture Details (v7.5)
- **RWKV-7 "Apex" Core**: Implemented State-Shift (x-shift) and Adaptive Time Decay for superior context retention and grammatical fluency.
- **Vectorized SNN-GNN**: Fully vectorized training pipeline achieving **<0.5s per epoch** on 500-sentence datasets.
- **Dynamic Adaptive Clustering**: Functional node partitioning (Concept, Social, Context) with learnable world-model states.
- **Spiking Efficiency**: Hardware-mimicking bypass logic that skips computation for inactive/silent neurons.
- **Crawl4AI Integration**: Automated knowledge ingestion from live web sources (Wikipedia).

---

## Future Roadmap: Phase 4 (Infinite Dynamic Nodes) 🌌

The next major evolution will shift from "Fixed Nodes" to "Emergent Nodes", similar to how Graphify operates.

### 1. Dynamic Node Expansion (Non-Fixed Population)
- **Problem**: Currently, `MODEL_NODES` is a fixed hyperparameter.
- **Goal**: Implement an architecture where nodes are "born" when new concepts are discovered. Using **Growing Neural Gas (GNG)** or **Dynamic Self-Organizing Maps** logic inside the GNN.

### 2. Autonomous Community Detection
- **Goal**: Instead of fixed cluster counts (e.g., 4), the model will use **Louvain-style or Spectral Clustering** in the latent space to determine the natural number of communities on-the-fly.

### 3. Sparse Loss & Entropy Constraints
- **Goal**: Force the model to maintain distinct, high-entropy knowledge clusters to prevent "Concept Bleeding" in large datasets (100k+ nodes).

### 4. BPE Tokenization Integration
- **Goal**: Replace simple word-level tokenization with native Byte-Pair Encoding (BPE) for better handling of Hinglish slangs and multi-lingual nuances.

---

## Technical Targets
- [x] Training Speed < 2s/epoch (Current: 0.3s)
- [x] Biological Spiking & GNN Interaction
- [x] RWKV-7 Adaptive State-Steering
- [ ] Infinite Node Growth (Phase 4)
- [ ] 100k+ Node Scalability Test
- [ ] Distributed Training Support
