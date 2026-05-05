"""
Batch Training + Chat Evaluation
- 10 batches with 2 epochs each
- Chat evaluation after each batch
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
)
from src.chat import PersistentGNNBrain

BATCH_SIZE = 50
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
                    print(f'  Error: {e}')
                    continue
        return results
    except ImportError:
        print('crawl4ai not installed, using fallback...')
        return await fallback_fetch(urls)

async def fallback_fetch(urls):
    import requests
    from bs4 import BeautifulSoup
    results = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for s in soup(['script', 'style']):
                s.decompose()
            text = soup.get_text()
            cleaned = clean_text(text)
            if len(cleaned) > 100:
                results.append(cleaned)
        except Exception as e:
            print(f'  Error: {e}')
            continue
    return results

async def get_search_results(query, num_results=10):
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        results = [r['href'] for r in ddgs.text(query, max_results=num_results)]
        return results
    except Exception as e:
        print(f'Search unavailable: {e}')
        return []

def train_on_batch(text_data, model, w2i, i2w, vocab_size, epochs=2):
    combined_text = '\n'.join(text_data).lower()
    data, new_w2i, new_i2w = get_data_from_text(combined_text)

    if w2i is None:
        merged_w2i = new_w2i.copy()
    else:
        merged_w2i = w2i.copy()
        for word in new_w2i:
            if word not in merged_w2i:
                merged_w2i[word] = len(merged_w2i)

    merged_i2w = {i: w for w, i in merged_w2i.items()}
    vocab_size = len(merged_w2i)

    if vocab_size > model.word_emb.num_embeddings:
        model.word_emb = torch.nn.Embedding(vocab_size, MODEL_DIM)
        model.head = torch.nn.Linear(MODEL_DIM, vocab_size)

    input_ids = [merged_w2i[new_i2w[idx]] for idx in data if idx in new_i2w]

    if len(input_ids) < 50:
        print('  Not enough data for training')
        return w2i, vocab_size

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

        denom = max(1, len(input_ids) // chunk_size)
        print(f'    Epoch {epoch+1}/{epochs} | Loss: {total_loss/denom:.4f}')

    return merged_w2i, vocab_size

async def main():
    print('='*60)
    print('WEB CRAWL + TRAIN + CHAT LOOP')
    print(f'Batches: {NUM_BATCHES}, Epochs per batch: {EPOCHS}')
    print('='*60)

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
        'artificial intelligence basics',
        'machine learning tutorials',
        'python programming examples',
        'neural networks explained',
        'deep learning concepts',
        'natural language processing',
        'data science introduction',
        'computer science fundamentals',
        'web development guide',
        'cloud computing basics',
    ]

    for batch_num in range(1, NUM_BATCHES + 1):
        print(f'')
        print(f'{"="*50}')
        print(f'BATCH {batch_num}/{NUM_BATCHES}')
        print(f'{"="*50}')

        topic = topics[(batch_num - 1) % len(topics)]
        print(f'[1] Searching: {topic}')

        urls = await get_search_results(topic, num_results=BATCH_SIZE)

        if not urls:
            wiki_topic = topic.replace(' ', '_')
            urls = [f'https://en.wikipedia.org/wiki/{wiki_topic}']

        print(f'   Found {len(urls)} URLs')

        print(f'[2] Crawling {len(urls)} URLs...')
        text_data = await crawl_urls(urls[:BATCH_SIZE])

        if not text_data:
            print('No content crawled, skipping...')
            continue

        print(f'   Crawled {len(text_data)} pages')

        print(f'[3] Training on batch {batch_num}...')
        w2i, vocab_size = train_on_batch(text_data, model, w2i, i2w, vocab_size, epochs=EPOCHS)

        torch.save(model.state_dict(), model_path)
        vocab_data = {'w2i': w2i, 'i2w': {str(k): v for k, v in i2w.items()}}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f)

        print(f'[4] Evaluating after batch {batch_num}...')

        brain = PersistentGNNBrain()

        test_queries = [
            'hello how are you?',
            'what is artificial intelligence?',
            'explain machine learning',
            'tell me about neural networks',
        ]

        print('')
        print('   Chat Evaluation:')
        for q in test_queries:
            resp = brain.chat(q, max_len=30)
            print(f'   Q: {q}')
            print(f'   A: {resp[:100]}')
            print('')

        print(f'   Batch {batch_num} complete!')

    print('')
    print('='*60)
    print('TRAINING COMPLETE - All 10 batches finished!')
    print('='*60)

if __name__ == '__main__':
    asyncio.run(main())