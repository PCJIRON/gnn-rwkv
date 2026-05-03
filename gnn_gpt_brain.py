import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import collections
import sys

# Ensure terminal supports UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class BilingualGNNBrain:
    def __init__(self, story_path):
        print("--- Initializing Final Bilingual GNN Brain ---")
        self.nlp_spacy = spacy.load("en_core_web_sm")
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        model_name = 'google/flan-t5-small'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        with open(story_path, 'r', encoding='utf-8') as f:
            self.story_text = f.read()
            
        # Secret Language Mapping (Aether-Lex)
        self.cipher_bridge = {
            "Kael": "Vance",
            "vosh": "control",
            "drax": "fleet",
            "Zorph": "King"
        }
        
        # Logical Entity Mapping for GNN
        self.logical_entities = {
            "Arthur": ["Arthur", "आर्थर", "Zorph"],
            "Vance": ["Vance", "वेंस", "Kael"],
            "Genevieve": ["Genevieve", "जेनेवीव"],
            "Silas": ["Silas", "साइलस"],
            "Maya": ["Maya", "माया"]
        }
        
        self.story_sentences = [s.strip() for s in self.story_text.split("\n") if s.strip()]
        self.influence_scores = {}
        self._build_brain()

    def _build_brain(self):
        print("Building Social Graph & Language Bridge...")
        entity_connections = collections.defaultdict(list)
        for i, sent in enumerate(self.story_sentences):
            for entity, aliases in self.logical_entities.items():
                if any(alias in sent for alias in aliases):
                    entity_connections[entity].append(i)
        
        # Calculate social influence based on graph connectivity
        entity_list = list(self.logical_entities.keys())
        for i, e in enumerate(entity_list):
            self.influence_scores[e] = len(entity_connections[e]) + 1.0
        
        print(f"Brain Ready. Mapped {len(entity_list)} entities across languages and ciphers.")

    def chat(self, query):
        # 1. Identify target character in query
        target_entity = None
        for entity, aliases in self.logical_entities.items():
            if any(alias.lower() in query.lower() for alias in aliases) or entity.lower() in query.lower():
                target_entity = entity
                break
        
        # 2. Build Context (including secret language knowledge)
        context_sentences = []
        if target_entity:
            for alias in self.logical_entities[target_entity]:
                context_sentences.extend([s for s in self.story_sentences if alias in s])
        
        # Add secret language tips to context if detected
        bridge_tips = ""
        for cipher, real in self.cipher_bridge.items():
            if cipher.lower() in query.lower():
                bridge_tips += f" (Note: {cipher} means {real})"
        
        context = " ".join(list(set(context_sentences))) if context_sentences else self.story_text
        context += bridge_tips
        
        # 3. Generate Answer
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer correctly using context:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Brain: {answer}")

if __name__ == "__main__":
    # Test internal logic
    brain = BilingualGNNBrain("story.txt")
    brain.chat("Who is Vance?")
    brain.chat("Zorph कौन है?")
