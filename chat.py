import torch
import torch.nn.functional as F
import json
import os
import sys
import asyncio
import subprocess
import re
from gnn_rwkv_story_gen import SocialNarrativeModel, get_data
from crawl4ai import AsyncWebCrawler

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class PersistentGNNBrain:
    def __init__(self, weights_path, vocab_path):
        self.weights_path = weights_path
        self.vocab_path = vocab_path
        self.load_brain()

    def load_brain(self):
        print(f"--- Loading GNN-RWKV Brain from {self.weights_path} ---")
        if not os.path.exists(self.vocab_path):
            print("Vocab file not found. Please train first.")
            return

        with open(self.vocab_path, "r", encoding='utf-8') as f:
            v_data = json.load(f)
            self.w2i = v_data["w2i"]
            self.i2w = {int(k): v for k, v in v_data["i2w"].items()}
        
        self.vocab_size = len(self.w2i)
        self.num_nodes = 5
        self.dim = 64
        
        self.model = SocialNarrativeModel(self.vocab_size, self.num_nodes, self.dim)
        if os.path.exists(self.weights_path):
            self.model.load_state_dict(torch.load(self.weights_path))
        self.model.eval()
        
        self.edges = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 1]], dtype=torch.long)
        self.node_ids = torch.tensor([0, 1, 2, 3, 4])
        self.h = torch.zeros((self.num_nodes, self.dim))

    def chat(self, user_input):
        import re
        clean_input = re.sub(r'[^\w\s]', '', user_input.lower())
        tokens = clean_input.split()
        if not tokens: return "..."
        
        current_id = None
        for word in tokens:
            if word in self.w2i:
                current_id = self.w2i[word]
                break
        
        if current_id is None:
            return "Mujhe ye topic abhi nahi pata, please /train command use karein."
        
        response = []
        with torch.no_grad():
            for _ in range(15):
                word_tensor = torch.tensor([current_id])
                logits, self.h = self.model(word_tensor, self.node_ids, self.edges, self.h)
                probs = F.softmax(logits[0] / 0.7, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                word = self.i2w[next_id]
                if word == ".": break
                response.append(word)
                current_id = next_id
        return " ".join(response)

def get_search_results(query):
    print(f"Searching for '{query}' using ddgs CLI...")
    try:
        # Run ddgs CLI and capture output, ignoring decoding errors
        result = subprocess.run(
            ['ddgs', 'text', '-k', query, '-m', '1'],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        output = result.stdout
        
        # Extract URL using Regex
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', output)
        if urls:
            return urls[0], output
        return None, output
    except Exception as e:
        print(f"CLI Search Error: {e}")
        return None, None

async def crawl_and_train(query_or_url, brain):
    url = None
    search_text = ""
    
    if query_or_url.startswith("http"):
        url = query_or_url
    else:
        url, search_text = get_search_results(query_or_url)
        
    if not url:
        # Fallback: Manual Wiki URL
        wiki_topic = query_or_url.replace(" ", "_").title()
        url = f"https://en.wikipedia.org/wiki/{wiki_topic}"
        print(f"No direct link found. Trying Smart Fallback: {url}")

    print(f"Attempting to learn from: {url}")
    content = ""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            content = result.markdown
    except Exception as e:
        print(f"Crawling failed: {e}")
        if search_text:
            print("Using search snippet as fallback data...")
            content = search_text
        else:
            print("Aborting. No data found.")
            return

    temp_file = "crawled_data.txt"
    with open(temp_file, "w", encoding='utf-8') as f:
        f.write(content)
    
    epochs = input("Kitne epochs pe train karna hai? (Default 10): ")
    epochs = int(epochs) if epochs.strip() else 10
    
    print(f"Training Brain on new knowledge for {epochs} epochs...")
    from train import incremental_train
    incremental_train(temp_file, epochs=epochs)
    
    brain.load_brain()
    print("--- LIVE TRAINING COMPLETE ---")

async def start_chat():
    weights = "social_brain.pth"
    vocab = "vocab.json"
    
    if not os.path.exists(weights):
        print("Model weights not found. Running initial training on story.txt...")
        from train import train_new_data
        train_new_data("story.txt", 10)

    brain = PersistentGNNBrain(weights, vocab)
    print("\n" + "="*50)
    print("      LIVE LEARNING GNN-RWKV (DDGS CLI)")
    print("="*50)
    print("Commands: /train <topic/url> | quit")
    
    while True:
        try:
            u_input = input("\nYou: ").strip()
            if u_input.lower() in ['quit', 'exit']: break
            if u_input.startswith("/train"):
                query = u_input.replace("/train", "").strip()
                await crawl_and_train(query, brain)
                continue
            if not u_input: continue
            ans = brain.chat(u_input)
            print(f"Brain: {ans}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(start_chat())
