import asyncio
import os
import re
from crawl4ai import AsyncWebCrawler
from train import train_brain

def clean_wiki_text(markdown):
    # Remove markdown noise
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown) # links
    text = re.sub(r"[*_#>`|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    # Take meaningful sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 20 and not s.startswith(" ")]
    return "\n".join(cleaned[:300]) # Take top 300 sentences

async def main():
    url = "https://en.wikipedia.org/wiki/Human_evolution"
    print(f"--- Crawling Wikipedia: {url} ---")
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        content = clean_wiki_text(result.markdown)
    
    with open("wiki_crawled.txt", "w", encoding='utf-8') as f:
        f.write(content)
    
    print(f"--- Crawling Done. Training on {len(content.splitlines())} sentences ---")
    
    # Train for 5 epochs as requested
    train_brain("wiki_crawled.txt", epochs=5, max_sentences=300)
    print("--- Training Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
