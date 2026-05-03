import asyncio
import os
import torch
from crawl4ai import AsyncWebCrawler
from train import train_new_data
from chat import PersistentGNNBrain

async def run_wiki_test():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"--- 1. Crawling Wikipedia: {url} ---")
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        content = result.markdown
    
    # Save to temp file
    with open("wiki_ai.txt", "w", encoding='utf-8') as f:
        f.write(content[:10000]) # Limit to 10k chars for fast test
        
    print(f"--- 2. Training on Wikipedia Data (10 Epochs) ---")
    train_new_data("wiki_ai.txt", epochs=10)
    
    print(f"--- 3. Testing Brain with AI Knowledge ---")
    brain = PersistentGNNBrain("social_brain.pth", "vocab.json")
    
    query = "What is artificial intelligence ?"
    print(f"\nUser: {query}")
    answer = brain.chat(query)
    print(f"Brain: {answer}")

if __name__ == "__main__":
    asyncio.run(run_wiki_test())
