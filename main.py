"""
Graphify-NLP + RWKV-7 Pipeline Demo
====================================
Full pipeline:
  1. Build word graph from text corpus (Graphify-style)
  2. Compute RWKV-7 WKV values from graph (no training!)
  3. Generate text using graph-driven RWKV-7

Usage:
  python main.py                          # Run with built-in demo corpus
  python main.py --file corpus.txt        # Run with custom corpus file
  python main.py --prompt "the world"     # Generate from specific prompt
"""

import argparse
import glob
import os
import sys
import time
import json

from word_graph import WordGraphBuilder, build_word_graph
from graph_rwkv7 import GraphRWKV7, TextGenerator, GraphEmbedding, create_generator


# =============================================================================
# DEMO CORPUS — Hindi + English mixed (Hinglish) + Pure English
# =============================================================================

DEMO_CORPUS = [
    # English
    "The sun rises in the east and sets in the west. Every morning the birds sing beautiful songs.",
    "Knowledge is power. The more you learn, the more you grow. Education opens doors to the world.",
    "The river flows through the valley, carrying stories of ancient times. Mountains stand tall and proud.",
    "Music is the language of the soul. Every note carries emotion. Rhythm moves the heart and mind.",
    "Technology changes the world every day. Computers and phones connect people across the globe.",
    "The stars shine bright in the dark night sky. The moon watches over the sleeping world below.",
    "Books are windows to other worlds. Reading expands the mind and feeds the imagination.",
    "The forest is alive with the sound of nature. Trees whisper secrets to the wind.",
    "Love is the strongest force in the universe. It connects hearts across time and space.",
    "Science seeks to understand the mysteries of the universe. Every discovery reveals new questions.",
    "Art speaks when words fail. Colors and shapes express feelings that language cannot capture.",
    "The ocean is vast and deep, full of wonders and mysteries. Waves dance on the shore.",
    "Children are the future of the world. Their laughter fills the air with joy and hope.",
    "History teaches us lessons from the past. We learn from both victories and mistakes.",
    "The garden blooms with flowers of every color. Butterflies dance from petal to petal.",
    
    # Hinglish
    "duniya mein bahut kuch seekhne ko hai. har din ek naya mauka hai.",
    "zindagi ek safar hai. har mod par kuch naya milta hai.",
    "khushi chahiye to doosron ko khush karo. pyaar baatne se badhta hai.",
    "sapne dekhna band mat karo. mehnat karo aur sapne sach honge.",
    "dosti duniya ki sabse khoobsurat cheez hai. sachche dost hamesha saath rehte hain.",
    "prakriti mein har cheez ka ek makam hai. ped paudhe hawa paani sab zaroori hai.",
    "sangeet dil ki baat kehta hai. sur taal mein bahut takat hai.",
    "technology ne duniya badal di hai. computer phone sab connect karte hain.",
    "kitaabein gyaan ka bhandar hain. padhne se insaan ka dimag tez hota hai.",
    "samundar bahut gehra aur vistaar hai. leherein kinaare par nachti hain.",
]


def load_file(path):
    """Load a text file, handling special characters in filenames via glob."""
    # First try direct open
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except (FileNotFoundError, OSError):
        pass
    
    # Try glob pattern matching
    matches = glob.glob(path)
    if not matches:
        # Try with wildcard for tricky chars
        dirname = os.path.dirname(path) or '.'
        basename = os.path.basename(path)
        # Replace special chars with ?
        pattern = os.path.join(dirname, basename.replace('\u2019', '?').replace('\u2018', '?'))
        matches = glob.glob(pattern)
    
    if not matches:
        # Last resort: find by prefix
        dirname = os.path.dirname(path) or '.'
        prefix = os.path.basename(path)[:20]
        for f in os.listdir(dirname):
            if f.startswith(prefix):
                matches = [os.path.join(dirname, f)]
                break
    
    if matches:
        with open(matches[0], 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    raise FileNotFoundError(f"Cannot find file: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Graphify-NLP + RWKV-7: Training-free text generation from word graphs"
    )
    parser.add_argument("--file", type=str, help="Path to corpus text file (or glob pattern)")
    parser.add_argument("--prompt", type=str, default="", help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=15, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--grammar-weight", type=float, default=0.3, help="Grammar enforcement weight")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Co-occurrence window size")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum word frequency")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--export-graph", type=str, default=None, help="Export graph JSON to path")
    parser.add_argument("--report", action="store_true", help="Print graph report")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--use-file", action="store_true", help="Auto-detect and use first .txt file in directory")
    
    args = parser.parse_args()
    
    # ===== LOAD CORPUS =====
    using_file = False
    if args.file:
        print(f"Loading corpus from: {args.file}")
        texts = load_file(args.file)
        using_file = True
    elif args.use_file:
        # Auto-detect first .txt file
        txt_files = [f for f in os.listdir('.') if f.endswith('.txt') and f != 'requirements.txt']
        if txt_files:
            chosen = txt_files[0]
            print(f"Auto-detected: {chosen}")
            texts = load_file(chosen)
            using_file = True
        else:
            print("No .txt files found, using built-in demo corpus")
            texts = DEMO_CORPUS
    else:
        print("Using built-in demo corpus (Hindi + English)")
        texts = DEMO_CORPUS
    
    print(f"   {len(texts)} text segments loaded\n")
    
    # ===== BUILD WORD GRAPH (Graphify-style) =====
    print("🔨 Building word graph (Graphify-NLP)...")
    t0 = time.time()
    
    builder = build_word_graph(texts, window_size=args.window, min_freq=args.min_freq)
    
    t1 = time.time()
    print(f"   ✅ Graph built in {t1-t0:.2f}s")
    print(f"   📊 Vocab: {len(builder.vocab)} words")
    print(f"   🔗 Nodes: {builder.graph.number_of_nodes()}")
    print(f"   ➡️  Edges: {builder.graph.number_of_edges()}")
    
    # ===== GOD NODES (Graphify feature) =====
    print(f"\n🏆 God Nodes (Hub Words):")
    god_nodes = builder.get_god_nodes(10)
    for i, (word, score) in enumerate(god_nodes):
        freq = builder.word_freq[word]
        pos = builder.word_to_pos.get(word, "?")
        print(f"   {i+1}. '{word}' (PageRank: {score:.4f}, freq: {freq}, POS: {pos})")
    
    # ===== GRAPH REPORT =====
    if args.report:
        print("\n" + "="*60)
        report = builder.generate_report()
        print(report)
        print("="*60)
    
    # ===== EXPORT GRAPH =====
    if args.export_graph:
        builder.export_json(args.export_graph)
        print(f"\n💾 Graph exported to: {args.export_graph}")
    
    # ===== COMPUTE RWKV-7 WKV VALUES FROM GRAPH =====
    print(f"\n⚡ Computing RWKV-7 WKV values from graph structure (NO TRAINING!)...")
    t2 = time.time()
    
    generator = TextGenerator(builder, dim=args.dim)
    
    t3 = time.time()
    print(f"   ✅ WKV values computed in {t3-t2:.2f}s")
    print(f"   🧠 State matrix: [{args.dim} × {args.dim}]")
    print(f"   📐 R, W, K, V, A, G precomputed for {len(builder.vocab)} words")
    
    # ===== GENERATE TEXT =====
    if args.interactive:
        interactive_mode(generator, args)
    else:
        single_generation(generator, args, using_file, builder)


def single_generation(generator, args, using_file=False, builder=None):
    """Generate text from a prompt or default."""
    print(f"\n{'='*60}")
    print(f"TEXT GENERATION")
    print(f"{'='*60}")
    
    if args.prompt:
        prompts = [args.prompt]
    elif using_file and builder:
        # Auto-detect relevant prompts from god nodes and frequent content words
        god_nodes = builder.get_god_nodes(30)
        content_words = [
            w for w, score in god_nodes 
            if builder.word_to_pos.get(w, 'NOUN') in ('NOUN', 'VERB', 'ADJ')
            and len(w) > 2
        ]
        prompts = []
        for w in content_words[:3]:
            prompts.append(w)
        top_bigrams = sorted(builder.bigram_freq.items(), key=lambda x: x[1], reverse=True)
        bigram_prompts_added = 0
        for (w1, w2), freq in top_bigrams:
            if bigram_prompts_added >= 4:
                break
            if (w1 in builder.vocab and w2 in builder.vocab and
                len(w1) > 2 and len(w2) > 2 and
                builder.word_to_pos.get(w1) != 'PUNCT' and
                builder.word_to_pos.get(w2) != 'PUNCT'):
                prompts.append(f"{w1} {w2}")
                bigram_prompts_added += 1
        if not prompts:
            prompts = ["the", "he said", "she was"]
    else:
        prompts = [
            "the world",
            "knowledge is",
            "duniya mein",
            "music",
            "the sun",
        ]
    
    for prompt in prompts:
        print(f"\n🎯 Prompt: '{prompt}'")
        
        result = generator.generate(
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            grammar_weight=args.grammar_weight,
            seed=args.seed,
        )
        
        print(f"📤 Output: {result}")
    
    print(f"\n{'='*60}")
    print("✅ Done! No training was needed — all from graph structure.")
    print(f"{'='*60}")


def interactive_mode(generator, args):
    """Interactive chat mode."""
    print(f"\n{'='*60}")
    print(f"💬 INTERACTIVE MODE (type 'quit' to exit)")
    print(f"{'='*60}")
    
    while True:
        try:
            prompt = input("\n🎯 You: ").strip()
            if prompt.lower() in {'quit', 'exit', 'q'}:
                print("👋 Bye!")
                break
            
            if not prompt:
                continue
            
            result = generator.generate(
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                grammar_weight=args.grammar_weight,
            )
            
            print(f"🤖 Bot: {result}")
            
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break


if __name__ == "__main__":
    main()
