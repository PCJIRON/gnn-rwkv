from gnn_story_brain import GNNStoryBrain

def run_autonomous_full_test():
    print("=========================================")
    print("AUTONOMOUS TRAINING & TESTING CYCLE")
    print("=========================================")
    
    # 1. Training (Automatic Extraction from new story)
    print("\n[PHASE 1] AUTOMATED TRAINING (STORY -> GNN)")
    brain = GNNStoryBrain("story.txt")
    
    # 2. Testing (Social Reasoning Chat)
    print("\n[PHASE 2] AUTOMATED CHAT TESTING (GPT-GENERATION)")
    
    queries = [
        "Who is the most powerful person in the Iron Coast?",
        "How is Maya connected to Elena and Julian?",
        "What is the relationship between Vance and Thorne?",
        "Who holds the secrets of the star-maps?",
        "Why is Thorne's influence fading?"
    ]
    
    for q in queries:
        brain.chat(q)

    print("\n=========================================")
    print("TEST CYCLE COMPLETE")
    print("=========================================")

if __name__ == "__main__":
    run_autonomous_full_test()
