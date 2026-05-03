# Roadmap: Human-Centric Dynamic GNN

This roadmap outlines the path from research to a functional simulation of dynamic human relationships using GNNs.

## Milestone 1: Environment & Foundation
- [x] **Phase 1.1: Core Infrastructure (UV Managed)**
  - Initialize project with `uv`.
  - Setup PyTorch, PyTorch Geometric, and Sentence-Transformers via `uv add`.
- [ ] **Phase 1.2: Dynamic Event System**
  - Implement an event queue for adding/removing nodes and edges.
  - Create a "Timeline" manager to track graph state over time.

## Milestone 2: Model Architecture
- [ ] **Phase 2.1: Heterogeneous GNN Implementation**
  - Build relational layers to handle typed edges (Family, Friends).
  - Implement Graph Attention (GAT) to learn influence weights.
- [ ] **Phase 2.2: Temporal Integration**
  - Integrate a Temporal Graph Network (TGN) memory module.
  - Enable the model to "remember" past relationships.

## Milestone 3: Simulation & Visualization
- [ ] **Phase 3.1: Human Behavior Simulation**
  - Create a script where nodes autonomously form and break links based on simple rules.
- [ ] **Phase 3.2: Influence Analysis Dashboard**
  - Visualize "Power Centers" in the graph.
  - Export "Influence Rankings" as nodes move through the network.
- [ ] **Phase 3.3: NLP Task Integration (Leader Prediction)**
  - Encode human node "Bios" using NLP (Sentence-Transformers).
  - Train the GNN to predict "Leadership Potential" based on social links + text features.

## Milestone 4: Advanced Evolution (Automated Brain)
- [ ] **Phase 4.1: Story-to-GNN Extraction**
  - Implement NER-based node discovery from raw English text.
  - Automate relationship detection via co-occurrence analysis.
- [ ] **Phase 4.2: Generative Chat Integration**
  - Link GNN influence scores to a Transformer (GPT) prompt.
  - Enable conversational Q&A about character power and social dynamics.
