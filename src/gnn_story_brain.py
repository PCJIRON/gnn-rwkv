import spacy
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nlp_gnn_demo import GNNInfluenceModel
import collections

class GNNStoryBrain:
    def __init__(self, story_path):
        print("--- Initializing GNN Story Brain ---")
        self.nlp_spacy = spacy.load("en_core_web_sm")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator = pipeline('text-generation', model='distilgpt2')
        
        with open(story_path, 'r') as f:
            self.story_text = f.read()
            
        self.entities = []
        self.relationships = []
        self.influence_scores = {}
        
        self._extract_graph()
        self._build_and_train_gnn()

    def _extract_graph(self):
        print("Extracting nodes and relationships from text...")
        doc = self.nlp_spacy(self.story_text)
        
        # 1. Extract PERSON entities as Nodes
        persons = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
        self.entities = list(persons)
        print(f"Nodes Found: {self.entities}")
        
        # 2. Extract Relationships via Co-occurrence in sentences
        entity_to_idx = {name: i for i, name in enumerate(self.entities)}
        edges = []
        
        for sent in doc.sents:
            found_in_sent = [ent.text for ent in sent.ents if ent.text in persons]
            if len(found_in_sent) > 1:
                for i in range(len(found_in_sent)):
                    for j in range(i + 1, len(found_in_sent)):
                        u, v = entity_to_idx[found_in_sent[i]], entity_to_idx[found_in_sent[j]]
                        edges.append([u, v])
                        edges.append([v, u]) # Undirected
        
        if not edges:
            # Fallback if no co-occurrence found (unlikely in this story)
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _build_and_train_gnn(self):
        print("Building and Training GNN for Social Analysis...")
        # Features from character context in story
        node_features = []
        for person in self.entities:
            # Find sentences mentioning this person as "bio/context"
            context = " ".join([sent.text for sent in self.nlp_spacy(self.story_text).sents if person in sent.text])
            node_features.append(self.embed_model.encode(context))
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create Dummy Training Target based on Degree (Simple heuristic for demo)
        # In a real system, this would be trained on actual outcomes
        degrees = collections.Counter(self.edge_index[0].tolist())
        y = torch.tensor([[degrees[i]/10.0] for i in range(len(self.entities))], dtype=torch.float)
        
        data = Data(x=x, edge_index=self.edge_index, y=y)
        
        # Train GNN
        model = GNNInfluenceModel(input_dim=x.shape[1], hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = model(data)
            for i, name in enumerate(self.entities):
                self.influence_scores[name] = preds[i].item()

    def chat(self, user_query):
        print(f"\nUser: {user_query}")
        
        # Build a more detailed social context
        top_chars = sorted(self.influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        social_context = " | ".join([f"{n} (Influence: {s:.2f})" for n, s in top_chars])
        
        prompt = (
            f"Background: This is a social network analysis. "
            f"The social hierarchy is: {social_context}. "
            f"Question about the story: {user_query} "
            f"Short Analysis:"
        )
        
        # Using repetition_penalty and top_k for better sentence variety
        response = self.generator(
            prompt, 
            max_new_tokens=40, 
            num_return_sequences=1, 
            repetition_penalty=1.5,
            do_sample=True,
            top_k=50,
            truncation=True
        )
        
        text = response[0]['generated_text']
        answer = text.split("Short Analysis:")[-1].strip()
        
        # Basic cleanup of GPT artifacts
        clean_answer = answer.split("\n")[0]
        print(f"Brain: {clean_answer}")

if __name__ == "__main__":
    brain = GNNStoryBrain("story.txt")
    
    # Simulate a Generative Chat
    brain.chat("Who are the most powerful people in Eldoria?")
    brain.chat("Tell me about the relationship between Arthur and Silas.")
    brain.chat("What do you think about Leo?")
