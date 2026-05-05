import torch
from chat import PersistentGNNBrain

def test_big_brain():
    weights = "social_brain.pth"
    vocab = "vocab.json"
    
    brain = PersistentGNNBrain(weights, vocab)
    
    questions = [
        "bhai, mera dil toot gaya hai aaj.",
        "kya tum mere liye ek app bana sakte ho?",
        "agli baar milenge toh kya khayenge?",
        "duniya mein sabse zyada kya zaroori hai?",
        "mummy gussa kar rahi hain ghar pe."
    ]
    
    print("\n" + "="*50)
    print("      BIG BRAIN HINGLISH TEST RESULTS")
    print("="*50)
    
    for q in questions:
        print(f"\nYou: {q}")
        ans = brain.chat(q, temp=0.7)
        print(f"Brain: {ans}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_big_brain()
