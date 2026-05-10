import sys
import os
import torch

sys.stdout.reconfigure(encoding='utf-8')
from token_graph import TokenGraph
from graph_rwkv7_v3 import InfiniteContextGeneratorV3

def main():
    print("="*70)
    print("  Graph-RWKV Infinite Context CLI (V3 Architecture)")
    print("="*70)
    
    graph_path = 'graphify-out/v3_graph.json'
    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found. Please run 'add_hinglish.py' first.")
        sys.exit(1)
        
    print(f"Loading Graph Brain from {graph_path}...")
    try:
        graph = TokenGraph.load(graph_path)
        session = InfiniteContextGeneratorV3(graph, n_heads=4)
        print(f"Graph loaded successfully! (Vocab size: {graph.vocab_size} tokens)")
        print("Ready! Type 'exit' or 'quit' to close.")
        print("Type '/reset' to clear conversation memory.")
        print("="*70)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        sys.exit(1)
    
    while True:
        try:
            user_input = input("\n[You]: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if user_input.lower() == '/reset':
                session.soft_reset()
                print("--- Context memory wiped ---")
                continue
                
            if not user_input:
                continue
                
            # Generate response
            response = session.generate(prompt=user_input, max_tokens=30, temperature=0.7, seed=None)
            
            # The generate function returns the prompt + generated text. We slice it.
            if response.startswith(user_input):
                bot_response = response[len(user_input):].strip()
            else:
                bot_response = response
                
            print(f"[Bot]: {bot_response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
