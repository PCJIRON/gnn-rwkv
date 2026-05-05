"""
Fast Batch Training + Chat Evaluation
- 10 batches with 2 epochs each
- 20 URLs per batch for speed
"""

import asyncio
import json
import os
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from src.gnn_rwkv_story_gen import (
    SocialNarrativeModel,
    MODEL_DIM, MODEL_NODES, MODEL_LAYERS, MODEL_CONTEXT,
    get_data_from_text,
    TOKENIZER_VERSION,
)
from src.chat import PersistentGNNBrain

BATCH_SIZE = 20
EPOCHS = 2
NUM_BATCHES = 10

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    return text.strip()

async def crawl_urls(urls):
    try:
        from crawl4ai import AsyncWebCrawler
        results = []
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url)
                    if result.success:
                        content = clean_text(result.markdown or result.text or '')
                        if len(content) > 100:
                            results.append(content)
                except Exception as e:
                    continue
        return results
    except:
        return await fallback_fetch(urls)

async def fallback_fetch(urls):
    import requests
    from bs4 import BeautifulSoup
    results = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for s in soup(['script', 'style']):
                s.decompose()
            text = soup.get_text()
            cleaned = clean_text(text)
            if len(cleaned) > 100:
                results.append(cleaned)
        except:
            continue
    return results

async def get_search_results(query, num_results=10):
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        results = [r['href'] for r in ddgs.text(query, max_results=num_results)]
        return results
    except:
        wiki_topic = query.replace(' ', '_')
        return [f'https://en.wikipedia.org/wiki/{wiki_topic}']

def train_on_batch(text_data, model, w2i, i2w, vocab_size, epochs=2):
    combined_text = ' '.join(text_data[:10]).lower()  # Use first 10 for speed

    try:
        data, new_w2i, new_i2w = get_data_from_text(combined_text)
    except:
        return w2i, vocab_size

    if w2i is None:
        merged_w2i = new_w2i.copy()
    else:
        merged_w2i = w2i.copy()
        for word in new_w2i:
            if word not in merged_w2i:
                merged_w2i[word] = len(merged_w2i)

    merged_i2w = {i: w for w, i in merged_w2i.items()}
    vocab_size = len(merged_w2i)

    try:
        if vocab_size > model.word_emb.num_embeddings:
            model.word_emb = torch.nn.Embedding(vocab_size, MODEL_DIM)
            model.head = torch.nn.Linear(MODEL_DIM, vocab_size)
    except:
        pass

    try:
        input_ids = [merged_w2i[new_i2w[idx]] for idx in data if idx in new_i2w]
    except:
        input_ids = []

    if len(input_ids) < 30:
        print('  Not enough data')
        return w2i, vocab_size

    device = 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    node_ids = torch.arange(MODEL_NODES, device=device, dtype=torch.long)

    chunk_size = 8
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        h = [torch.zeros((MODEL_NODES, MODEL_DIM), device=device) for _ in range(MODEL_LAYERS)]

        for i in range(0, min(len(input_ids) - chunk_size, 200), chunk_size):  # Limit to 200 chunks for speed
            chunk = torch.tensor(input_ids[i:i+chunk_size], device=device, dtype=torch.long)
            target = torch.tensor(input_ids[i+1:i+chunk_size+1], device=device, dtype=torch.long)

            optimizer.zero_grad()
            try:
                logits, h = model(chunk, node_ids, None, h)
                h = [s.detach() for s in h]
                loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except:
                continue

        denom = max(1, min(len(input_ids) // chunk_size, 200))
        print(f'    Epoch {epoch+1}/{epochs} | Loss: {total_loss/denom:.4f}')

    return merged_w2i, vocab_size

async def main():
    print('='*50)
    print('FAST BATCH TRAINING - 10 Batches x 2 Epochs')
    print('='*50)

    model_path = 'social_brain.pth'
    vocab_path = 'vocab.json'

    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print('Loading existing model...')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            w2i = vocab_data.get('w2i', {})
        vocab_size = len(w2i)
        i2w = {int(k): v for k, v in vocab_data.get('i2w', {}).items()}

        model = SocialNarrativeModel(vocab_size, MODEL_NODES, MODEL_DIM, MODEL_LAYERS, MODEL_CONTEXT)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print('Initializing new model...')
        w2i = {}
        vocab_size = 1000
        model = SocialNarrativeModel(vocab_size, MODEL_NODES, MODEL_DIM, MODEL_LAYERS, MODEL_CONTEXT)
        i2w = {}

    topics = [
        'artificial intelligence',
        'machine learning',
        'python programming',
        'neural networks',
        'deep learning',
        'natural language processing',
        'data science',
        'computer science',
        'web development',
        'cloud computing',
    ]

    start_batch = 1
    if os.path.exists('batch_progress.txt'):
        with open('batch_progress.txt', 'r') as f:
            start_batch = int(f.read().strip()) + 1

    for batch_num in range(start_batch, NUM_BATCHES + 1):
        print(f'')
        print(f'=== BATCH {batch_num}/{NUM_BATCHES} ===')

        topic = topics[(batch_num - 1) % len(topics)]
        print(f'[Crawl] {topic}')

        urls = await get_search_results(topic, num_results=BATCH_SIZE)
        print(f'  Found {len(urls)} URLs')

        text_data = await crawl_urls(urls[:BATCH_SIZE])
        print(f'  Crawled {len(text_data)} pages')

        if text_data:
            print(f'[Train] Batch {batch_num}...')
            w2i, vocab_size = train_on_batch(text_data, model, w2i, i2w, vocab_size, epochs=EPOCHS)

            i2w = {v: k for k, v in w2i.items()}
            torch.save(model.state_dict(), model_path)
            vocab_data = {'w2i': w2i, 'i2w': {str(k): v for k, v in i2w.items()}}
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f)

            print(f'[Chat] Evaluating after batch {batch_num}...')
            brain = PersistentGNNBrain(weights_path=model_path, vocab_path=vocab_path)

            test_queries = [
                'hello how are you?',
                'what is ai?',
                'explain machine learning',
            ]

            for q in test_queries:
                resp = brain.chat(q, max_len=25)
                print(f'  Q: {q}')
                print(f'  A: {resp[:80]}')

            print(f'✓ Batch {batch_num} complete')
            with open('batch_progress.txt', 'w') as f:
                f.write(str(batch_num))
        else:
            print(f'  No data, skipping...')

    print('')
    print('='*50)
    print('ALL 10 BATCHES COMPLETE!')
    print('='*50)

if __name__ == '__main__':
    asyncio.run(main())