import torch
from chat import PersistentGNNBrain

def test_big_brain():
    weights = "social_brain.pth"
    vocab = "vocab.json"
    
    brain = PersistentGNNBrain(weights, vocab)
    
    questions = [
        "aur bhai kya haal hai",
        "aaj bahut traffic hai",
        "khana kha liya"
    ]
    
    print("\n" + "="*50)
    print("      BIG BRAIN HINGLISH TEST RESULTS")
    print("="*50)
    
    for q in questions:
        print(f"\nYou: {q}")
        ans = brain.chat(q)
        print(f"Brain: {ans}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_big_brain()
