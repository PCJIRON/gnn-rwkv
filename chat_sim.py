import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import sys

# Re-using the model from demo
from nlp_gnn_demo import GNNInfluenceModel, get_nlp_model

class SocialGraphChat:
    def __init__(self):
        print("--- System Initializing ---")
        self.nlp = get_nlp_model()
        self.humans = [
            {"name": "Alice", "bio": "Expert in AI and social networking."},
            {"name": "Bob", "bio": "Student of digital ethics and data science."},
            {"name": "Charlie", "bio": "Business strategist and venture capitalist."},
            {"name": "Diana", "bio": "Human rights activist and community organizer."}
        ]
        self.edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        self.model = None
        self.data = None
        self._prepare_data()
        self._train_initial_model()

    def _prepare_data(self):
        bios = [h['name'] + ": " + h['bio'] for h in self.humans]
        node_features = self.nlp.encode(bios)
        x = torch.tensor(node_features, dtype=torch.float)
        # Dummy targets for training
        y = torch.tensor([[0.8], [0.3], [0.9], [0.5]], dtype=torch.float)
        self.data = Data(x=x, edge_index=self.edge_index, y=y)

    def _train_initial_model(self):
        print("Training social influence model on English data...")
        self.model = GNNInfluenceModel(input_dim=self.data.x.shape[1], hidden_dim=16)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        self.model.train()
        for _ in range(50):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = criterion(out, self.data.y)
            loss.backward()
            optimizer.step()
        print("Model Ready for Chat!\n")

    def get_influence(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
        return out

    def add_person(self, name, bio):
        print(f"Integrating {name} into the neural network...")
        # 1. Update list
        self.humans.append({"name": name, "bio": bio})
        
        # 2. Update Features
        new_bio_embed = self.nlp.encode([name + ": " + bio])
        self.data.x = torch.cat([self.data.x, torch.tensor(new_bio_embed, dtype=torch.float)], dim=0)
        
        # 3. New nodes start unconnected (or we could auto-connect)
        print(f"{name} is now a node. Use 'connect' to build relationships.")

    def connect(self, name1, name2):
        try:
            idx1 = next(i for i, h in enumerate(self.humans) if h['name'].lower() == name1.lower())
            idx2 = next(i for i, h in enumerate(self.humans) if h['name'].lower() == name2.lower())
            
            new_edges = torch.tensor([[idx1, idx2], [idx2, idx1]], dtype=torch.long)
            self.data.edge_index = torch.cat([self.data.edge_index, new_edges], dim=1)
            print(f"Relationship established between {self.humans[idx1]['name']} and {self.humans[idx2]['name']}.")
        except StopIteration:
            print("Error: One or both names not found.")

    def run_chat(self):
        print("--- GNN SOCIAL SIMULATOR ---")
        print("Commands: list, query <name>, add <name> <bio>, connect <name1> <name2>, exit")
        
        while True:
            cmd_input = input("\nYou > ").strip().split(" ", 2)
            if not cmd_input: continue
            
            cmd = cmd_input[0].lower()
            
            if cmd == "exit":
                break
            elif cmd == "list":
                inf = self.get_influence()
                for i, h in enumerate(self.humans):
                    print(f"- {h['name']}: {h['bio'][:40]}... (Influence: {inf[i].item():.4f})")
            elif cmd == "query" and len(cmd_input) > 1:
                name = cmd_input[1]
                inf = self.get_influence()
                try:
                    idx = next(i for i, h in enumerate(self.humans) if h['name'].lower() == name.lower())
                    print(f"Node: {self.humans[idx]['name']}")
                    print(f"Bio: {self.humans[idx]['bio']}")
                    print(f"Predicted Influence: {inf[idx].item():.4f}")
                except StopIteration:
                    print(f"Person '{name}' not found.")
            elif cmd == "add" and len(cmd_input) > 2:
                name = cmd_input[1]
                bio = cmd_input[2]
                self.add_person(name, bio)
            elif cmd == "connect" and len(cmd_input) > 2:
                names = cmd_input[1:]
                # Handle spaces if names are separated by space
                parts = cmd_input[2].split(" ", 1)
                if len(parts) > 0:
                    self.connect(cmd_input[1], parts[0])
            else:
                print("Unknown command or missing arguments.")

if __name__ == "__main__":
    chat = SocialGraphChat()
    chat.run_chat()
