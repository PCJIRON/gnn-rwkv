# Project: Human-Centric Dynamic GNN (HCD-GNN)

## Vision
To model human social behavior and relationships using Dynamic Graph Neural Networks (DGNNs). In this system, individuals are nodes, relationships (typed edges like family, friends) represent connectivity, and "influence power" is a learned property derived from the network's topology and evolution.

## Core Concepts
- **Nodes (Humans)**: Carry personal state, features, and internal neural representations.
- **Typed Edges (Relationships)**: Categorized connections (Father, Mother, Friend, etc.) in a heterogeneous graph.
- **Dynamic Topology**: Nodes can join/leave, and relationships can form/dissolve at any time.
- **Influence Power**: A metric (e.g., centrality or attention-weighted score) indicating a node's ability to affect its neighborhood.

## Technical Goals
1. Implement a **Dynamic Heterogeneous Graph** structure.
2. Develop an **Inductive GNN** that can handle new nodes and edges without retraining.
3. Integrate a **Temporal Mechanism** (e.g., TGN or DySAT) to model the evolution of relationships over time.
4. Calculate **Influence Scores** based on the dynamic graph structure.
5. Use **UV** for lighting-fast Python dependency and environment management.
