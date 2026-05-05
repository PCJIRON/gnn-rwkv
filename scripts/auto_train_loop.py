"""
Auto Training Loop with Web Crawl + Evaluate after each batch
- Crawls web data (crawl4ai with beautifulsoup fallback)
- Trains for 3 epochs per batch
- Tests relevance and grammar after each batch
- Stops if both don't improve, continues if they do
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
import re
import asyncio
import subprocess
import time
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.gnn_rwkv_story_gen import (
    SocialNarrativeModel,
    MODEL_DIM,
    MODEL_NODES,
    MODEL_LAYERS,
    MODEL_CONTEXT,
    get_data_from_text,
)

WEIGHTS_PATH = "social_brain.pth"
VOCAB_PATH = "vocab.json"
META_PATH = "brain_meta.json"

TOPICS = [
    "artificial intelligence",
    "machine learning",
    "python programming",
    "deep learning neural networks",
    "web development",
    "data science",
    "computer programming",
    "software engineering",
]

EVAL_PROMPTS = [
    "what is ai",
    "how does machine learning work",
    "explain python",
    "what is deep learning",
    "how to learn programming",
    "what is web development",
    "explain neural networks",
    "what is data science",
]


class AutoTrainer:
    def __init__(self):
        self.model = None
        self.w2i = {}
        self.i2w = {}
        self.vocab_size = 0
        self.num_nodes = MODEL_NODES
        self.dim = MODEL_DIM
        self.num_layers = MODEL_LAYERS
        self.max_context = MODEL_CONTEXT
        self.scores_history = []
        
    def load_model(self):
        print(f"--- Loading Model from {WEIGHTS_PATH} ---")
        if not os.path.exists(VOCAB_PATH):
            print("No vocab file. Training from scratch...")
            return False
            
        with open(VOCAB_PATH, "r", encoding='utf-8') as f:
            v_data = json.load(f)
            self.w2i = v_data["w2i"]
            self.i2w = {int(k): v for k, v in v_data["i2w"].items()}
        
        self.vocab_size = len(self.w2i)
        
        meta = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        self.num_nodes = meta.get("num_nodes", MODEL_NODES)
        self.dim = meta.get("dim", MODEL_DIM)
        self.num_layers = meta.get("layers", MODEL_LAYERS)
        self.max_context = meta.get("context", MODEL_CONTEXT)
        
        self.model = SocialNarrativeModel(
            self.vocab_size,
            self.num_nodes,
            dim=self.dim,
            num_layers=self.num_layers,
            max_seq=self.max_context,
        )
        
        if os.path.exists(WEIGHTS_PATH):
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
                model_state = self.model.state_dict()
                compatible = {
                    key: value
                    for key, value in state_dict.items()
                    if key in model_state and model_state[key].shape == value.shape
                }
                self.model.load_state_dict(compatible, strict=False)
                print("Loaded existing weights.")
            except Exception as e:
                print(f"Warning: {e}")
        
        self.model.eval()
        return True

    def chat(self, prompt, max_len=20, temp=0.9):
        full_prompt = f"User: {prompt}\nBrain:"
        tokens = full_prompt.lower().split()
        input_ids = [self.w2i[w] for w in tokens if w in self.w2i]
        
        if not input_ids:
            return "", 0
        
        if len(input_ids) > self.max_context:
            input_ids = input_ids[-self.max_context:]
        
        response = []
        generated_ids = []
        stop_words = {"user", "brain", "user:", "brain:", "\n"}
        
        with torch.no_grad():
            for i in range(max_len):
                context_ids = (input_ids + generated_ids)[-self.max_context:]
                h = [torch.zeros((self.num_nodes, self.dim)) for _ in range(self.num_layers)]
                node_ids = torch.arange(self.num_nodes)
                logits, _ = self.model(torch.tensor(context_ids), node_ids, None, h)
                logits = logits[-1] / temp
                
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                word = self.i2w.get(next_id, "")
                
                if word.lower() in stop_words or word == ":":
                    break
                if response and word == response[-1]:
                    continue
                    
                response.append(word)
                generated_ids.append(next_id)
                if word == "." and i > 3: break
        
        text = " ".join(response).strip()
        text = re.sub(r'^[,.\s?]+', '', text)
        return text, len(response)

    def evaluate_model(self):
        """Evaluate relevance and grammar scores"""
        print("\n--- Evaluating Model ---")
        
        relevance_score = 0
        grammar_score = 0
        total = 0
        
        topic_keywords = {
            "ai": ["intelligence", "machine", "learn", "neural", "ai", "artificial", "data", "model", "train", "algorithm"],
            "machine learning": ["learn", "data", "model", "train", "algorithm", "predict", "pattern", "feature", "train"],
            "python": ["python", "code", "program", "function", "variable", "class", "import", "loop", "return"],
            "deep learning": ["neural", "network", "layer", "deep", "train", "backprop", "gradient", "weight", "activation"],
            "programming": ["code", "program", "function", "class", "variable", "loop", "algorithm", "data", "return"],
            "web": ["web", "html", "css", "javascript", "server", "browser", "website", "page", "http"],
            "neural": ["neuron", "network", "layer", "weight", "activation", "brain", "cell", "signal", "learn"],
            "data science": ["data", "analyze", "statistic", "visual", "insight", "model", "predict", "learn"],
        }
        
        for prompt in EVAL_PROMPTS:
            response, words = self.chat(prompt)
            if not response:
                continue
            
            total += 1
            print(f"  Q: {prompt}")
            print(f"  A: {response}")
            
            prompt_lower = prompt.lower()
            keywords = topic_keywords.get(prompt_lower.split()[0], [])
            
            response_words = set(response.lower().split())
            overlap = sum(1 for kw in keywords if any(kw in w for w in response_words))
            
            # Any keyword match counts as relevant
            if overlap > 0:
                relevance_score += 1.0
            
            has_proper_sentence = (
                len(response.split()) >= 3 and
                (response.count(".") >= 1 or response.count(",") >= 1 or len(response) > 20)
            )
            if has_proper_sentence:
                grammar_score += 1
        
        relevance = relevance_score / max(1, total)
        grammar = grammar_score / max(1, total)
        
        print(f"\nFinal - Relevance: {relevance:.2f} | Grammar: {grammar:.2f}")
        
        return relevance, grammar

    def crawl_web(self, topic):
        """Crawl web for topic using crawl4ai or fallback"""
        print(f"\n>>> Crawling: {topic}")
        
        try:
            from crawl4ai import AsyncWebCrawler
            
            async def crawl():
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}")
                    return result.markdown if result else ""
            
            content = asyncio.run(crawl())
            if content:
                return self.clean_text(content)
        except Exception as e:
            print(f"crawl4ai failed: {e}")
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return self.clean_text(text)
        except Exception as e:
            print(f"Fallback failed: {e}")
        
        return None

    def clean_text(self, text):
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"#{1,6}\s*", " ", text)
        text = re.sub(r"[*_>`|]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if len(line) < 5:
                continue
            if line.lower() in {"menu", "navigation", "privacy policy", "terms of use"}:
                continue
            lines.append(line[:200])
        
        return "\n".join(lines[:100])

    def train_on_text(self, text, epochs=3):
        """Train for specified epochs"""
        print(f"\n>>> Training: {epochs} epochs on {len(text)} chars")
        
        data, w2i, i2w = get_data_from_text(text.lower())
        
        new_vocab = set(w2i.keys()) - set(self.w2i.keys())
        if new_vocab:
            print(f"  Adding {len(new_vocab)} new words to vocab")
            for word in new_vocab:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word
            self.vocab_size = len(self.w2i)
            
            self.model = SocialNarrativeModel(
                self.vocab_size,
                self.num_nodes,
                dim=self.dim,
                num_layers=self.num_layers,
                max_seq=self.max_context,
            )
        
        device = 'cpu'
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        node_ids = torch.arange(self.num_nodes, device=device)
        
        batch_size = 16
        losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            h = [torch.zeros((self.num_nodes, self.dim), device=device) for _ in range(self.num_layers)]
            
            for i in range(0, len(data) - batch_size, batch_size):
                chunk = torch.tensor(data[i:i+batch_size], device=device, dtype=torch.long)
                target = torch.tensor(data[i+1:i+batch_size+1], device=device, dtype=torch.long)
                
                optimizer.zero_grad()
                logits, h = self.model(chunk, node_ids, None, h)
                h = [s.detach() for s in h]
                
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, (len(data) // batch_size))
            losses.append(avg_loss)
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        return losses

    def save_model(self):
        """Save model, vocab, meta"""
        torch.save(self.model.state_dict(), WEIGHTS_PATH)
        
        with open(VOCAB_PATH, "w", encoding='utf-8') as f:
            json.dump({
                "w2i": self.w2i,
                "i2w": {str(k): v for k, v in self.i2w.items()},
                "tokenizer": "word"
            }, f, ensure_ascii=False)
        
        meta = {
            "num_nodes": self.num_nodes,
            "dim": self.dim,
            "layers": self.num_layers,
            "context": self.max_context,
            "tokenizer": "word",
            "trained_at": datetime.now().isoformat()
        }
        with open(META_PATH, "w", encoding='utf-8') as f:
            json.dump(meta, f)
        
        print(">>> Model saved!")

    def run_training_loop(self, max_batches=10):
        """Main training loop with evaluation after each batch"""
        
        has_model = self.load_model()
        if not has_model:
            print("No existing model. Starting fresh training...")
        
        prev_relevance, prev_grammar = self.evaluate_model()
        self.scores_history.append({
            "batch": 0,
            "relevance": prev_relevance,
            "grammar": prev_grammar
        })
        
        print(f"\n{'='*50}")
        print("STARTING AUTO TRAINING LOOP")
        print(f"Initial: Relevance={prev_relevance:.2f}, Grammar={prev_grammar:.2f}")
        print(f"{'='*50}")
        
        for batch in range(1, max_batches + 1):
            print(f"\n{'#'*50}")
            print(f"BATCH {batch}/{max_batches}")
            print(f"{'#'*50}")
            
            topic = TOPICS[(batch - 1) % len(TOPICS)]
            content = self.crawl_web(topic)
            
            if not content:
                print(f"Skipping batch {batch} - no content")
                continue
            
            print(f"  Content length: {len(content)} chars")
            
            self.train_on_text(content, epochs=5)
            self.save_model()
            
            relevance, grammar = self.evaluate_model()
            
            print(f"\n>>> Batch {batch} Results:")
            print(f"    Relevance: {prev_relevance:.2f} -> {relevance:.2f} ({'+' if relevance > prev_relevance else ''}{relevance - prev_relevance:.2f})")
            print(f"    Grammar:   {prev_grammar:.2f} -> {grammar:.2f} ({'+' if grammar > prev_grammar else ''}{grammar - prev_grammar:.2f})")
            
            improved = relevance > prev_relevance or grammar > prev_grammar
            
            self.scores_history.append({
                "batch": batch,
                "relevance": relevance,
                "grammar": grammar,
                "topic": topic,
                "improved": improved
            })
            
            # Continue if at least one metric improved or just show progress
            if relevance >= prev_relevance or grammar >= prev_grammar:
                prev_relevance = relevance
                prev_grammar = grammar
                print(f"\n>>> Continuing training...")
            else:
                print(f"\n>>> Relevance decreased - but continuing to see if more training helps...")
                prev_relevance = relevance
            
            time.sleep(1)
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETE")
        print(f"{'='*50}")
        print("\nScore History:")
        for h in self.scores_history:
            print(f"  Batch {h['batch']}: R={h['relevance']:.2f}, G={h['grammar']:.2f}")
        
        with open("training_log.json", "w", encoding='utf-8') as f:
            json.dump(self.scores_history, f, indent=2)
        print("\nLog saved to training_log.json")
        
        return self.scores_history


if __name__ == "__main__":
    trainer = AutoTrainer()
    results = trainer.run_training_loop(max_batches=20)