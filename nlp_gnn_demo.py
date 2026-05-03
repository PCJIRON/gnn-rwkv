import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. NLP Node Feature Generation
def get_nlp_model():
    print("Initializing NLP model (Sentence-Transformers)...")
    return SentenceTransformer('all-MiniLM-L6-v2')

# Human nodes with text-based "Bios" or "States"
# 4. Define the GNN Model
class GNNInfluenceModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNInfluenceModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1) # Output a single influence score

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second layer to output prediction
        x = self.conv2(x, edge_index)
        return x

if __name__ == "__main__":
    nlp_model = get_nlp_model()
    
    humans = [
        {"name": "Alice", "bio": "Technology leader and community builder."},
        {"name": "Bob", "bio": "Software engineer interested in graph theory."},
        {"name": "Charlie", "bio": "Social media influencer and marketing expert."},
        {"name": "David", "bio": "New student learning about neural networks."},
        {"name": "Eve", "bio": "Experienced researcher in social sciences."}
    ]

    # Encode bios into vector embeddings (Node Features)
    bios = [h['name'] + ": " + h['bio'] for h in humans]
    node_features = nlp_model.encode(bios)
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. Relationship Definition (Edge Index)
    # Pairs represent [Source, Target] - e.g., Alice (0) is connected to Bob (1)
    # 0: Alice, 1: Bob, 2: Charlie, 3: David, 4: Eve
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 2, 3, 4, 0], # Source
        [1, 0, 2, 0, 2, 1, 3, 2, 0, 4]  # Target (Symmetric for undirected)
    ], dtype=torch.long)

    # 3. Targets: "Influence Score" (Dummy ground truth for demo)
    # Let's say Alice (0) and Charlie (2) have high influence
    y = torch.tensor([0.9, 0.4, 0.8, 0.2, 0.5], dtype=torch.float).view(-1, 1)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # 5. Training Loop (Demo)
    print("\nStarting Training Demo...")
    model = GNNInfluenceModel(input_dim=x.shape[1], hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    # 6. Inference & Dynamic Update Simulation
    model.eval()
    with torch.no_grad():
        pred = model(data)
        print("\nInitial Influence Predictions:")
        for i, h in enumerate(humans):
            print(f"{h['name']}: Predicted Score = {pred[i].item():.4f} (Actual: {y[i].item()})")

    # Simulate adding a new node (Frank) and a new connection
    print("\n--- DYNAMIC UPDATE: Adding Frank ---")
    new_human = {"name": "Frank", "bio": "High-power CEO and investor."}
    new_bio_embedding = nlp_model.encode([new_human['name'] + ": " + new_human['bio']])
    new_x = torch.cat([data.x, torch.tensor(new_bio_embedding, dtype=torch.float)], dim=0)

    # Connect Frank (5) to Alice (0) and Charlie (2)
    new_edges = torch.tensor([[5, 0, 5, 2], [0, 5, 2, 5]], dtype=torch.long)
    new_edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    new_data = Data(x=new_x, edge_index=new_edge_index)

    with torch.no_grad():
        new_pred = model(new_data)
        print(f"Frank (New Node) Predicted Influence: {new_pred[5].item():.4f}")
        print(f"Alice (Connected to Frank) New Influence: {new_pred[0].item():.4f}")
