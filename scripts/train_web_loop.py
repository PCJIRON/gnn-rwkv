"""
Web Crawl + Train + Evaluate Loop
- Fetch data using crawl4ai in batches
- Train on each batch
- Evaluate relevance and grammar after each batch
- Stop if performance drops
"""

import asyncio
import json
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F

from src.gnn_rwkv_story_gen import (
    SocialNarrativeModel,
    MODEL_DIM, MODEL_NODES, MODEL_LAYERS, MODEL_CONTEXT,
    get_data_from_text,
)
from src.chat import PersistentGNNBrain


# Config
BATCH_SIZE = 50  # URLs per batch
MAX_TRAINING_EPOCHS_PER_BATCH = 10
MIN_IMPROVEMENT = 0.05  # 5% minimum improvement required
TRAINING_DATA_DIR = Path("web_data")
TRAINING_DATA_DIR.mkdir(exist_ok=True)


def clean_text(text):
    """Clean crawled text for training"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special chars but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    return text.strip()


async def crawl_urls(urls):
    """Crawl URLs using crawl4ai"""
    try:
        from crawl4ai import Crawler
        
        crawler = Crawler()
        results = []
        
        for url in urls:
            try:
                result = await crawler.arun(url)
                if result.success:
                    content = clean_text(result.markdown or result.text or "")
                    if len(content) > 100:  # Only keep meaningful content
                        results.append(content)
            except Exception as e:
                print(f"  Error crawling {url}: {e}")
                continue
        
        return results
    except ImportError:
        print("crawl4ai not installed. Using fallback web fetch...")
        return await fallback_fetch(urls)


async def fallback_fetch(urls):
    """Fallback if crawl4ai not available"""
    import requests
    from bs4 import BeautifulSoup
    
    results = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Remove script and style
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            cleaned = clean_text(text)
            if len(cleaned) > 100:
                results.append(cleaned)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    return results


async def get_search_results(query, num_results=10):
    """Get search results using DuckDuckGo or fallback"""
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        results = [r['href'] for r in ddgs.text(query, max_results=num_results)]
        return results
    except Exception as e:
        print(f"Search unavailable: {e}")
        return []


def train_on_batch(text_data, model, w2i, i2w, vocab_size, epochs=10):
    """Train model on a batch of text"""
    # Combine all text
    combined_text = "\n".join(text_data).lower()
    
    # Tokenize
    data, new_w2i, new_i2w = get_data_from_text(combined_text)
    
    # Merge vocabulary - handle empty w2i
    if w2i is None:
        merged_w2i = new_w2i.copy()
    else:
        merged_w2i = w2i.copy()
        for word in new_w2i:
            if word not in merged_w2i:
                merged_w2i[word] = len(merged_w2i)
    
    merged_i2w = {i: w for w, i in merged_w2i.items()}
    vocab_size = len(merged_w2i)
    
    # Resize model if needed
    if vocab_size > model.word_emb.num_embeddings:
        model.word_emb = torch.nn.Embedding(vocab_size, MODEL_DIM)
        model.head = torch.nn.Linear(MODEL_DIM, vocab_size)
    
    # Prepare training data
    input_ids = [merged_w2i[new_i2w[idx]] for idx in data if idx in new_i2w]
    
    if len(input_ids) < 50:
        print("  Not enough data for training")
        return w2i, vocab_size
    
    # Training
    device = 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    node_ids = torch.arange(MODEL_NODES, device=device, dtype=torch.long)
    
    chunk_size = 16
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        h = [torch.zeros((MODEL_NODES, MODEL_DIM), device=device) for _ in range(MODEL_LAYERS)]
        
        for i in range(0, len(input_ids) - chunk_size, chunk_size):
            chunk = torch.tensor(input_ids[i:i+chunk_size], device=device, dtype=torch.long)
            target = torch.tensor(input_ids[i+1:i+chunk_size+1], device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            logits, h = model(chunk, node_ids, None, h)
            h = [s.detach() for s in h]
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/max(1,len(input_ids)//chunk_size):.4f}")
    
    return merged_w2i, vocab_size


def evaluate_model(model, brain, w2i, i2w):
    """Evaluate relevance and grammar"""
    
    test_queries = [
        # Simple factual
        ("what is artificial intelligence?", "ai,machine,learning,computer"),
        # Programming
        ("how to write python code?", "python,code,function,program"),
        # Science
        ("explain neural networks", "neural,network,neuron,brain"),
        # General
        ("what is the capital of india?", "delhi,india,capital,city"),
    ]
    
    relevance_scores = []
    grammar_scores = []
    
    for query, expected_keywords in test_queries:
        try:
            # PersistentGNNBrain uses 'max_len' instead of 'max_tokens'
            response = brain.chat(query, max_len=50)
            response_lower = response.lower()
            
            # Relevance: check if response contains expected keywords
            keywords = expected_keywords.split(',')
            hits = sum(1 for kw in keywords if kw in response_lower)
            relevance = hits / len(keywords)
            relevance_scores.append(relevance)
            
            # Grammar: simple checks
            grammar_score = 1.0
            
            # Check sentence structure
            sentences = re.split(r'[.!?]+', response)
            valid_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
            if sentences and len(sentences) > 0:
                grammar_score *= (valid_sentences / len(sentences))
            
            # Check for common errors
            has_double_space = '  ' in response
            has_missing_punctuation = not any(c in response for c in '.!?')
            
            if has_double_space:
                grammar_score *= 0.8
            if has_missing_punctuation:
                grammar_score *= 0.9
            
            grammar_scores.append(grammar_score)
            
        except Exception as e:
            print(f"  Eval error for '{query}': {e}")
            relevance_scores.append(0)
            grammar_scores.append(0)
    
    avg_relevance = sum(relevance_scores) / max(1, len(relevance_scores))
    avg_grammar = sum(grammar_scores) / max(1, len(grammar_scores))
    
    return avg_relevance, avg_grammar


async def main():
    print("="*60)
    print("WEB CRAWL + TRAIN + EVALUATE LOOP")
    print("="*60)
    
    # Initialize or load model
    model_path = "social_brain.pth"
    vocab_path = "vocab.json"
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print("\nLoading existing model...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            w2i = vocab_data.get('w2i', {})
        vocab_size = len(w2i)
        i2w = {int(k): v for k, v in vocab_data.get('i2w', {}).items()}
        
        model = SocialNarrativeModel(vocab_size, MODEL_NODES, MODEL_DIM, MODEL_LAYERS, MODEL_CONTEXT)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("\nInitializing new model...")
        w2i = {}
        vocab_size = 1000
        model = SocialNarrativeModel(vocab_size, MODEL_NODES, MODEL_DIM, MODEL_LAYERS, MODEL_CONTEXT)
        i2w = {}
    
    # Initialize brain for chat
    brain = None
    if os.path.exists(model_path):
        try:
            brain = PersistentGNNBrain()
        except:
            brain = None
    
    # Baseline evaluation
    print("\n### Baseline Evaluation ###")
    baseline_relevance, baseline_grammar = 0.5, 0.5
    if brain and w2i:
        baseline_relevance, baseline_grammar = evaluate_model(model, brain, w2i, i2w)
        print(f"Baseline - Relevance: {baseline_relevance:.2f}, Grammar: {baseline_grammar:.2f}")
    
    # Training topics to crawl
    topics = [
        "artificial intelligence basics",
        "machine learning tutorials",
        "python programming examples",
        "neural networks explained",
        "deep learning concepts",
        "natural language processing",
        "data science introduction",
        "computer science fundamentals",
    ]
    
    current_relevance = baseline_relevance
    current_grammar = baseline_grammar
    batch_num = 0
    
    print(f"\nStarting training loop...")
    print(f"Target: {BATCH_SIZE} URLs per batch, {MAX_TRAINING_EPOCHS_PER_BATCH} epochs per batch")
    print(f"Min improvement required: {MIN_IMPROVEMENT*100}%")
    
    while batch_num < 5:  # Max 5 batches
        batch_num += 1
        print(f"\n{'='*50}")
        print(f"BATCH {batch_num}")
        print(f"{'='*50}")
        
        # Step 1: Get URLs for current topic
        topic = topics[(batch_num - 1) % len(topics)]
        print(f"\n[1] Searching for: {topic}")
        
        urls = await get_search_results(topic, num_results=BATCH_SIZE)
        
        if not urls:
            print("No URLs found, using fallback URLs...")
            urls = [
                f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                f"https://www.tutorialspoint.com/{topic.replace(' ', '_').split('_')[0]}/index.htm",
            ]
        
        print(f"   Found {len(urls)} URLs")
        
        # Step 2: Crawl URLs
        print(f"\n[2] Crawling {len(urls)} URLs...")
        text_data = await crawl_urls(urls[:BATCH_SIZE])
        
        if not text_data:
            print("No content crawled. Trying next topic...")
            continue
        
        print(f"   Crawled {len(text_data)} pages, {sum(len(t) for t in text_data)} chars")
        
        # Save batch data
        batch_file = TRAINING_DATA_DIR / f"batch_{batch_num}.txt"
        batch_file.write_text("\n".join(text_data), encoding='utf-8')
        
        # Step 3: Train on batch
        print(f"\n[3] Training on batch {batch_num}...")
        w2i, vocab_size = train_on_batch(
            text_data, model, w2i, i2w, vocab_size,
            epochs=MAX_TRAINING_EPOCHS_PER_BATCH
        )
        
        # Save model
        torch.save(model.state_dict(), model_path)
        vocab_data = {'w2i': w2i, 'i2w': {str(k): v for k, v in i2w.items()}}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f)
        
        # Update brain
        try:
            brain = PersistentGNNBrain()
        except:
            pass
        
        # Step 4: Evaluate
        print(f"\n[4] Evaluating after batch {batch_num}...")
        new_relevance, new_grammar = evaluate_model(model, brain, w2i, i2w)
        
        relevance_improvement = new_relevance - current_relevance
        grammar_improvement = new_grammar - current_grammar
        
        print(f"\n   Results:")
        print(f"   Relevance: {current_relevance:.2f} -> {new_relevance:.2f} ({relevance_improvement:+.2f})")
        print(f"   Grammar:   {current_grammar:.2f} -> {new_grammar:.2f} ({grammar_improvement:+.2f})")
        
        # Check if performance improved
        total_improvement = (relevance_improvement + grammar_improvement) / 2
        
        if total_improvement >= MIN_IMPROVEMENT:
            print(f"\n   [PASS] Improvement: {total_improvement:.2f} >= {MIN_IMPROVEMENT}")
            print("   Continuing to next batch...")
            current_relevance = new_relevance
            current_grammar = new_grammar
        else:
            print(f"\n   [STOP] Insufficient improvement: {total_improvement:.2f} < {MIN_IMPROVEMENT}")
            print("   Training stopped.")
            break
    
    # Final report
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total batches: {batch_num}")
    print(f"Final relevance: {current_relevance:.2f} (started: {baseline_relevance:.2f})")
    print(f"Final grammar: {current_grammar:.2f} (started: {baseline_grammar:.2f})")
    print(f"Model saved to: {model_path}")
    print(f"Vocab saved to: {vocab_path}")


if __name__ == "__main__":
    asyncio.run(main())