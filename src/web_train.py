import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from chat import PersistentGNNBrain, clean_crawled_text
from src.train import train_brain
from chat import load_active_brain_paths


DEFAULT_SEEDS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Reasoning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Java_(programming_language)",
]


def resolve_seed(seed):
    if seed.startswith("http://") or seed.startswith("https://"):
        return seed
    return f"https://en.wikipedia.org/wiki/{quote(seed.replace(' ', '_'))}"


def chunk_text(text, max_chars=25000):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def vocab_size_mb(path="vocab.json"):
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


EVAL_PROMPTS = [
    {
        "prompt": "Explain artificial intelligence in simple English.",
        "keywords": {"artificial", "intelligence", "systems", "human", "tasks"},
    },
    {
        "prompt": "What is machine learning?",
        "keywords": {"machine", "learning", "data", "patterns"},
    },
    {
        "prompt": "What is reasoning?",
        "keywords": {"reasoning", "think", "logic", "conclusion"},
    },
    {
        "prompt": "Write a tiny Java hello world program.",
        "keywords": {"java", "program", "hello", "class", "print"},
    },
]


def grammar_score(answer):
    words = re.findall(r"[a-zA-Z]+", answer)
    if len(words) < 4:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
    has_sentence_end = 1.0 if re.search(r"[.!?]$", answer.strip()) else 0.0
    bad_markers = len(re.findall(r"\b(user|brain|menu|navigation|retrieved|citation|edit)\b", answer.lower()))
    length_score = min(1.0, len(words) / 12)
    return max(0.0, (0.45 * unique_ratio) + (0.35 * length_score) + (0.2 * has_sentence_end) - (0.15 * bad_markers))


def relevance_score(answer, keywords):
    words = set(re.findall(r"[a-zA-Z]+", answer.lower()))
    if not keywords:
        return 0.0
    return len(words & keywords) / len(keywords)


def evaluate_chat(batch_index):
    weights, vocab, _ = load_active_brain_paths()
    brain = PersistentGNNBrain(weights, vocab)
    results = []
    grammar_scores = []
    relevance_scores = []
    print("\n--- Quality Probe ---")
    for item in EVAL_PROMPTS:
        prompt = item["prompt"]
        answer = brain.chat(prompt, max_len=28, temp=0.85, top_p=0.9)
        g_score = grammar_score(answer)
        r_score = relevance_score(answer, item["keywords"])
        grammar_scores.append(g_score)
        relevance_scores.append(r_score)
        results.append({
            "prompt": prompt,
            "answer": answer,
            "grammar": round(g_score, 3),
            "relevance": round(r_score, 3),
        })
        print(f"Q: {prompt}")
        print(f"A: {answer}")
        print(f"grammar={g_score:.3f} relevance={r_score:.3f}")

    max_context = getattr(brain.model, "max_seq", 128)
    context_results = []
    for length in [32, 64, 96, 128, 160, 256]:
        prompt = " ".join(["context"] * length)
        try:
            _ = brain.chat(prompt, max_len=4)
            ok = True
        except Exception as exc:
            ok = False
            results.append({"context_error": str(exc), "length": length})
        context_results.append({"input_words": length, "ok": ok, "effective_cap_tokens": max_context})

    record = {
        "batch_index": batch_index,
        "vocab_mb": vocab_size_mb(),
        "avg_grammar": round(sum(grammar_scores) / len(grammar_scores), 3),
        "avg_relevance": round(sum(relevance_scores) / len(relevance_scores), 3),
        "combined": round(((sum(grammar_scores) / len(grammar_scores)) + (sum(relevance_scores) / len(relevance_scores))) / 2, 3),
        "chat": results,
        "context": context_results,
    }
    with open("training_probe_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Context cap: last {max_context} tokens are used when input is longer.")
    return record


async def crawl_url(url):
    project_home = Path.cwd()
    (project_home / ".crawl4ai").mkdir(parents=True, exist_ok=True)
    (project_home / ".crawl4ai" / "robots").mkdir(parents=True, exist_ok=True)
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(project_home)

    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(base_directory=str(project_home)) as crawler:
        result = await crawler.arun(url=url)
    return clean_crawled_text(result.markdown)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="*", help="URLs or Wikipedia topics. Defaults to core English AI/code pages.")
    parser.add_argument("--epochs-per-batch", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means no explicit chunk limit.")
    parser.add_argument("--chunk-chars", type=int, default=25000)
    parser.add_argument("--probe-every", type=int, default=2)
    parser.add_argument("--vocab-limit-mb", type=float, default=100.0)
    parser.add_argument("--stop-if-no-improve", action="store_true", default=True)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    args = parser.parse_args()

    seeds = args.seeds or DEFAULT_SEEDS
    urls = [resolve_seed(seed) for seed in seeds[: args.max_pages]]
    state_path = Path("web_training_state.json")
    batch_file = Path("web_training_batch.txt")
    total_batches = 0
    best_score = None

    for page_index, url in enumerate(urls, start=1):
        if vocab_size_mb() >= args.vocab_limit_mb:
            print(f"Stopping: vocab.json is {vocab_size_mb():.2f} MB")
            break

        print(f"\n=== Crawling page {page_index}/{len(urls)}: {url} ===")
        try:
            text = await crawl_url(url)
        except Exception as exc:
            print(f"Crawl failed: {exc}")
            continue

        chunks = chunk_text(text, max_chars=args.chunk_chars)
        print(f"Clean text: {len(text):,} chars -> {len(chunks)} batch chunk(s)")

        for chunk_index, chunk in enumerate(chunks, start=1):
            if args.max_chunks and total_batches >= args.max_chunks:
                print("Reached max chunk limit for this run.")
                return
            if vocab_size_mb() >= args.vocab_limit_mb:
                print(f"Stopping: vocab.json is {vocab_size_mb():.2f} MB")
                return

            total_batches += 1
            batch_file.write_text(chunk, encoding="utf-8")
            print(f"\n--- Training batch {total_batches} (page {page_index}, chunk {chunk_index}) ---")
            result = train_brain(
                str(batch_file),
                epochs=args.epochs_per_batch,
                persist=True,
                reset_existing=False,
                verbose=True,
            )
            state = {
                "last_url": url,
                "page_index": page_index,
                "chunk_index": chunk_index,
                "total_batches": total_batches,
                "vocab_mb": vocab_size_mb(),
                "result": {
                    "vocab_size": result.get("vocab_size") if result else None,
                    "new_tokens": result.get("new_tokens") if result else None,
                    "reused_tokens": result.get("reused_tokens") if result else None,
                },
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"vocab.json size: {state['vocab_mb']:.2f} MB")

            if args.probe_every and total_batches % args.probe_every == 0:
                record = evaluate_chat(total_batches)
                score = record["combined"]
                if best_score is None:
                    best_score = score
                    print(f"Initial quality score: {score:.3f}")
                elif score >= best_score + args.min_improvement:
                    print(f"Quality improved: {best_score:.3f} -> {score:.3f}; continuing.")
                    best_score = score
                elif args.stop_if_no_improve:
                    print(
                        f"Stopping: quality did not improve enough. "
                        f"best={best_score:.3f}, current={score:.3f}, min_delta={args.min_improvement:.3f}"
                    )
                    return


if __name__ == "__main__":
    asyncio.run(main())
