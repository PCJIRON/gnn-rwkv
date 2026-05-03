from chat import PersistentGNNBrain
import sys

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def test_brain():
    weights = "social_brain.pth"
    vocab = "vocab.json"
    
    print("--- STARTING AUTONOMOUS HINGLISH CHAT TEST ---")
    brain = PersistentGNNBrain(weights, vocab)
    
    test_queries = [
        "hello",
        "kaise ho ?",
        "kya chal raha hai ?",
        "khana kha liya ?",
        "bye"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = brain.chat(query)
        print(f"Brain: {response}")

if __name__ == "__main__":
    test_brain()
