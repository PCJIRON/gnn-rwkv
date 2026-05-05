import random

def generate_chat_corpus(output_path, num_sentences=500):
    subjects = ["AI", "Python", "GNN", "RWKV", "Delhi", "India", "The brain", "Machine learning", "Data science", "The model"]
    verbs = ["is", "works by", "helps in", "improves", "analyzes", "creates", "learns from", "processes"]
    objects = ["knowledge", "context", "efficiency", "code", "neural networks", "social graphs", "information", "patterns"]
    
    questions = [
        "What is {}?",
        "How does {} work?",
        "Why is {} important?",
        "Tell me about {}.",
        "Explain {} in simple terms.",
        "Can {} learn?",
        "Where is {} used?",
        "Is {} fast on CPU?"
    ]
    
    answers = [
        "{} is a powerful tool for {}ing {}.",
        "{} works using {} to process {} effectively.",
        "{} is important because it {}s {} in real-time.",
        "{} is used in {} to build better {} systems.",
        "Yes, {} can {} {} very well.",
        "{} on CPU is optimized for {} and {}."
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_sentences // 2):
            subj = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects)
            
            q = random.choice(questions).format(subj)
            a = random.choice(answers).format(subj, verb[:-1] if verb.endswith("s") else verb, obj)
            
            f.write(f"User: {q}\n")
            f.write(f"Brain: {a} <eos>\n")

if __name__ == "__main__":
    generate_chat_corpus("chat_corpus_500.txt", 500)
    print("Generated chat_corpus_500.txt with 500 sentences.")
