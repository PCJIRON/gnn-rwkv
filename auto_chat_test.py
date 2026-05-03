from chat_sim import SocialGraphChat

def run_automated_test():
    print("--- Automated Chat & GNN Test ---")
    chat = SocialGraphChat()
    
    # 1. Initial State
    print("\n[Step 1] Initial Network State:")
    chat.run_chat_automated("list")
    
    # 2. Add new nodes
    print("\n[Step 2] User: add Frank High-power CEO and investor from Silicon Valley.")
    chat.add_person("Frank", "High-power CEO and investor from Silicon Valley.")
    
    print("\n[Step 3] User: add Sam Junior developer and eager learner.")
    chat.add_person("Sam", "Junior developer and eager learner.")
    
    # 3. Form Relationships
    print("\n[Step 4] User: connect Frank Alice")
    chat.connect("Frank", "Alice")
    
    print("\n[Step 5] User: connect Sam Alice")
    chat.connect("Sam", "Alice")
    
    # 4. Check Influence Evolution
    print("\n[Step 6] User: query Alice")
    inf = chat.get_influence()
    idx = next(i for i, h in enumerate(chat.humans) if h['name'] == "Alice")
    print(f"Alice's New Influence Score: {inf[idx].item():.4f}")
    
    print("\n[Step 7] User: list")
    inf = chat.get_influence()
    for i, h in enumerate(chat.humans):
        print(f"- {h['name']}: Influence = {inf[i].item():.4f}")

# Adding a helper to chat_sim for automation
import chat_sim
def run_chat_automated(self, cmd_string):
    inf = self.get_influence()
    for i, h in enumerate(self.humans):
        print(f"- {h['name']}: Influence = {inf[i].item():.4f}")

chat_sim.SocialGraphChat.run_chat_automated = run_chat_automated

if __name__ == "__main__":
    run_automated_test()
