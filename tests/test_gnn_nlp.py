import pytest
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from nlp_gnn_demo import GNNInfluenceModel # Import the model from the demo

@pytest.fixture
def nlp_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@pytest.fixture
def sample_data(nlp_model):
    bios = ["Hello, I am a leader.", "I am a follower."]
    node_features = nlp_model.encode(bios)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def test_nlp_embedding_shape(nlp_model):
    """Test if NLP model produces the expected embedding dimension (384 for MiniLM)."""
    embedding = nlp_model.encode(["Test bio"])
    assert embedding.shape == (1, 384)

def test_gnn_model_forward(sample_data):
    """Test if the GNN model produces a single output per node."""
    input_dim = sample_data.x.shape[1]
    model = GNNInfluenceModel(input_dim=input_dim, hidden_dim=16)
    out = model(sample_data)
    assert out.shape == (2, 1) # 2 nodes, 1 output each

def test_dynamic_node_addition(nlp_model, sample_data):
    """Test if we can dynamically add a node and its edges."""
    # Existing state
    x = sample_data.x
    edge_index = sample_data.edge_index
    
    # Add new node
    new_bio = nlp_model.encode(["High power CEO"])
    new_x = torch.cat([x, torch.tensor(new_bio, dtype=torch.float)], dim=0)
    
    # Add new edge (New node 2 connected to node 0)
    new_edges = torch.tensor([[2, 0], [0, 2]], dtype=torch.long)
    new_edge_index = torch.cat([edge_index, new_edges], dim=1)
    
    assert new_x.shape[0] == 3 # Should have 3 nodes now
    assert new_edge_index.shape[1] == 4 # Should have 4 directed edges now (2 original + 2 new)

def test_model_training_step(sample_data):
    """Test if a single training step reduces loss (basic sanity check)."""
    input_dim = sample_data.x.shape[1]
    model = GNNInfluenceModel(input_dim=input_dim, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    target = torch.tensor([[1.0], [0.0]], dtype=torch.float)
    
    # Initial loss
    out = model(sample_data)
    initial_loss = criterion(out, target)
    
    # Train step
    optimizer.zero_grad()
    out = model(sample_data)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    
    # Final loss should be different (usually smaller)
    out_new = model(sample_data)
    final_loss = criterion(out_new, target)
    assert final_loss != initial_loss
