import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import collections
import sys

# Ensure terminal supports UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class GNNStoryteller:
    def __init__(self, story_path):
        print("--- Initializing GNN Narrative Engine (Storyteller) ---")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Using GPT-2 for creative storytelling
        self.generator = pipeline('text-generation', model='gpt2')
        
        with open(story_path, 'r', encoding='utf-8') as f:
            self.story_text = f.read()
            
        self.influence_scores = {
            "Admiral Vance": 5.0, # High power
            "King Arthur": 4.5,
            "Duke Silas": 3.0,
            "Lady Genevieve": 2.5,
            "Maya": 2.0
        }
        
        print("Narrative Engine Ready. World Bible Loaded.")

    def generate_epic_story(self, theme):
        print(f"\nGenerating Epic Story on theme: {theme}...")
        
        # 1. Convert GNN Data into a "World Bible" string for GPT
        social_context = "\n".join([f"- {name} (Influence Power: {score})" for name, score in self.influence_scores.items()])
        
        # 2. Design the Storytelling Prompt
        prompt = (
            f"Write a long, detailed, and epic fantasy story based on the following social hierarchy:\n"
            f"{social_context}\n\n"
            f"The theme of the story is: {theme}\n"
            f"The story begins as follow:\n"
            f"In the ancient lands of the Iron Coast, a great shift in power was beginning. "
        )
        
        # 3. Generate a Long Story
        # We use high max_new_tokens and top_p for creative variety
        response = self.generator(
            prompt,
            max_new_tokens=250, # Making it long
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            truncation=True
        )
        
        story = response[0]['generated_text']
        
        print("\n" + "="*50)
        print("GENERATE STORY (BADI STORY):")
        print("="*50)
        print(story)
        print("="*50)

if __name__ == "__main__":
    teller = GNNStoryteller("story.txt")
    
    # Generate an epic story about rebellion and power
    teller.generate_epic_story("The Rebellion of the Iron Coast and the secret Aether-Lex code.")
