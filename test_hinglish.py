import sys
import torch
sys.stdout.reconfigure(encoding='utf-8')
from token_graph import TokenGraph
from graph_rwkv7_v3 import InfiniteContextGeneratorV3

print('='*70)
print('  TEST: Hinglish Everyday Conversations (V3 Infinite Context)')
print('='*70)

graph = TokenGraph.load('graphify-out/v3_graph.json')
gen = InfiniteContextGeneratorV3(graph, n_heads=4)

prompts = [
    "hello kaise ho",
    "kya kar rahe ho",
    "aaj mausam",
    "mujhe bhookh",
    "movie dekhne",
]

print("\n--- SINGLE PROMPTS ---")
for p in prompts:
    r = gen.generate(prompt=p, max_tokens=30, temperature=0.7, seed=42)
    print(f"  [{p}] -> {r}")

print("\n--- INFINITE CONTEXT SESSION (Chatting) ---")
session = InfiniteContextGeneratorV3(graph, n_heads=4)

chat_turns = [
    "hello bhai",
    "kahan jaa rahe ho?",
    "mere saath chaloge?",
    "thik hai chalo",
]

for turn in chat_turns:
    print(f"\n  User: {turn}")
    r = session.generate(prompt=turn, max_tokens=20, temperature=0.7, seed=42)
    # The output will include the prompt, let's just print the whole thing
    print(f"  Bot : {r}")
