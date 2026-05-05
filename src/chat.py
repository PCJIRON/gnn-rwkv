import torch
import torch.nn.functional as F
import json
import os
import sys
import asyncio
import subprocess
import re
from dataclasses import dataclass
from src.gnn_rwkv_story_gen import (
    SocialNarrativeModel,
    MODEL_DIM,
    MODEL_NODES,
    MODEL_LAYERS,
    MODEL_CONTEXT,
    get_data,
    get_data_from_text,
    TOKENIZER_VERSION,
    encode_text,
    decode_tokens,
)
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
except ImportError:
    PromptSession = None
    HTML = None

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

console = Console()
WEIGHTS_PATH = "social_brain.pth"
VOCAB_PATH = "vocab.json"
META_PATH = "brain_meta.json"
ACTIVE_BRAIN_PATH = "active_brain.json"
ROUTER_LOG_PATH = "router_trace.jsonl"
DEFAULT_DATA_FILE = "clean_chat_1000.txt"
DEFAULT_MAX_SENTENCES = 1000


@dataclass
class ChatMessage:
    role: str
    text: str


class PersistentGNNBrain:
    def __init__(self, weights_path=None, vocab_path=None, session_data=None):
        self.weights_path = weights_path if weights_path else WEIGHTS_PATH
        self.vocab_path = vocab_path if vocab_path else VOCAB_PATH
        if session_data:
            self.load_session_brain(session_data)
        else:
            self.load_brain()

    def load_session_brain(self, session_data):
        self.model = session_data["model"]
        self.w2i = session_data["w2i"]
        self.i2w = session_data["i2w"]
        self.vocab_size = session_data["vocab_size"]
        self.num_nodes = session_data["num_nodes"]
        self.dim = session_data["dim"]
        self.num_layers = session_data.get("layers", MODEL_LAYERS)
        self.tokenizer = session_data.get("tokenizer", TOKENIZER_VERSION)
        self.model.eval()
        self.node_ids = torch.arange(self.num_nodes)
        self.h = [torch.zeros((self.num_nodes, self.dim)) for _ in range(self.num_layers)]

    def load_brain(self):
        print(f"--- Loading GNN-RWKV Brain from {self.weights_path} ---")
        if not os.path.exists(self.vocab_path):
            print("Vocab file not found. Please train first.")
            return

        with open(self.vocab_path, "r", encoding='utf-8') as f:
            v_data = json.load(f)
            self.tokenizer = v_data.get("tokenizer", "word")
            self.w2i = v_data["w2i"]
            self.i2w = {int(k): v for k, v in v_data["i2w"].items()}
        
        self.vocab_size = len(self.w2i)
        meta = {}
        meta_path = META_PATH
        if os.path.exists(ACTIVE_BRAIN_PATH):
            try:
                with open(ACTIVE_BRAIN_PATH, "r", encoding="utf-8") as f:
                    meta_path = json.load(f).get("meta", META_PATH)
            except (OSError, json.JSONDecodeError):
                meta_path = META_PATH
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = {}
        self.num_nodes = meta.get("num_nodes", meta.get("nodes", MODEL_NODES))
        self.dim = meta.get("dim", MODEL_DIM)
        self.num_layers = meta.get("layers", MODEL_LAYERS)
        self.max_context = meta.get("context", MODEL_CONTEXT)
        
        self.model = SocialNarrativeModel(
            self.vocab_size,
            self.num_nodes,
            dim=self.dim,
            num_layers=self.num_layers,
            max_seq=self.max_context,
        )
        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location="cpu")
                model_state = self.model.state_dict()
                compatible = {
                    key: value
                    for key, value in state_dict.items()
                    if key in model_state and model_state[key].shape == value.shape
                }
                self.model.load_state_dict(compatible, strict=False)
                print("Successfully loaded Phase 3 weights.")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
        self.model.eval()
        
        self.node_ids = torch.arange(self.num_nodes)
        self.h = [torch.zeros((self.num_nodes, self.dim)) for _ in range(self.num_layers)]

    def chat(self, prompt, max_len=24, temp=0.9, top_p=0.9, rep_penalty=2.2):
        # 1. Clean and tokenize prompt with steering
        full_prompt = f"User: {prompt}\nBrain:"
        if self.tokenizer == TOKENIZER_VERSION:
            input_ids = encode_text(full_prompt, self.w2i)
        else:
            clean_prompt = full_prompt.lower().replace("?", " ?").replace(".", " .")
            tokens = clean_prompt.split()
            input_ids = [self.w2i[w] for w in tokens if w in self.w2i]
        
        if not input_ids:
            return "Hmm... mujhe ye samajh nahi aaya."

        max_context = getattr(self.model, "max_seq", 128)
        if len(input_ids) > max_context:
            input_ids = input_ids[-max_context:]
        
        print(f"[DEBUG] Input IDs: {input_ids}")
        print(f"[DEBUG] Decoded input: {decode_tokens([self.i2w[idx] for idx in input_ids])}")

        # 2. Build context
        response = []
        response_tokens = []
        generated_ids = []
        stop_words = {"user", "brain", "user:", "brain:", "\n", "<eos>", "<bos>", "<pad>"}
        
        with torch.no_grad():
            for i in range(max_len):
                context_ids = (input_ids + generated_ids)[-max_context:]
                h = [torch.zeros((self.num_nodes, self.dim)) for _ in range(self.num_layers)]
                logits, _ = self.model(torch.tensor(context_ids), self.node_ids, None, h)
                
                # Model returns [seq, vocab] - take last token
                if logits.dim() == 2:
                    logits = logits[-1]
                elif logits.dim() == 3:
                    logits = logits[0, -1]
                    
                logits = logits / temp
                
                # Penalty
                for seen_token in set(response_tokens):
                    if seen_token in self.w2i:
                        t_id = self.w2i[seen_token]
                        logits[t_id] -= rep_penalty
                
                # Top-P (Simple robust version)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Keep tokens with cumulative probability <= top_p
                keep_mask = cumulative_probs <= top_p
                keep_mask[0] = True # Always keep at least one
                
                indices_to_remove = sorted_indices[~keep_mask]
                logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                word = self.i2w[next_id]
                
                if word.lower() in stop_words:
                    break
                if word == ":":
                    break
                if response and word == response[-1]:
                    continue
                    
                response.append(word)
                response_tokens.append(word)
                generated_ids.append(next_id)
                # DEBUG PRINT
                print(f"[DEBUG] Generated token: {word}")
                if word == "." and i > 5: break
                
        # 4. Final Clean
        if self.tokenizer == TOKENIZER_VERSION:
            text = decode_tokens(response_tokens)
        else:
            text = " ".join(response).strip()
        text = re.sub(r'^[,.\s?]+', '', text)
        text = re.sub(r"\b(user|brain)\s*:\s*", "", text, flags=re.IGNORECASE)
        self._log_generation(prompt, getattr(self.model, "last_active_nodes", []), text)
        return text if text else "..."

    def _log_generation(self, prompt, active_nodes, response):
        record = {
            "query": prompt,
            "active_nodes": active_nodes,
            "response_preview": response[:200],
        }
        try:
            with open(ROUTER_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

def get_search_results(query):
    print(f"Searching for '{query}' using ddgs CLI...")
    try:
        # Run ddgs CLI and capture output, ignoring decoding errors
        result = subprocess.run(
            ['ddgs', 'text', '-k', query, '-m', '1'],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        output = result.stdout
        
        # Extract URL using Regex
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', output)
        if urls:
            return urls[0], output
        return None, output
    except Exception as e:
        print(f"CLI Search Error: {e}")
        return None, None

async def crawl_and_train(query_or_url, brain):
    url = None
    search_text = ""
    
    if query_or_url.startswith("http"):
        url = query_or_url
    else:
        url, search_text = get_search_results(query_or_url)
        
    if not url:
        # Fallback: Manual Wiki URL
        wiki_topic = query_or_url.replace(" ", "_").title()
        url = f"https://en.wikipedia.org/wiki/{wiki_topic}"
        print(f"No direct link found. Trying Smart Fallback: {url}")

    print(f"Attempting to learn from: {url}")
    content = ""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            content = clean_crawled_text(result.markdown)
    except Exception as e:
        print(f"Crawling failed: {e}")
        if search_text:
            print("Using search snippet as fallback data...")
            content = search_text
        else:
            print("Aborting. No data found.")
            return

    temp_file = "crawled_data.txt"
    with open(temp_file, "w", encoding='utf-8') as f:
        f.write(content)
    
    epochs = input("Kitne epochs pe train karna hai? (Default 10): ")
    epochs = int(epochs) if epochs.strip() else 10
    
    print(f"Training Brain on new knowledge for {epochs} epochs...")
    from train import train_brain

    train_brain(temp_file, epochs=epochs, persist=True, max_sentences=1000)
    brain.load_brain()
    print("--- LIVE TRAINING COMPLETE ---")


def clean_crawled_text(markdown):
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"#{1,6}\s*", " ", text)
    text = re.sub(r"[*_>`|]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 3:
            continue
        if line.lower() in {"menu", "navigation", "privacy policy", "terms of use"}:
            continue
        lines.append(line)
    return "\n".join(lines)


def render_chat(messages, status):
    body = []
    body.append(Panel(status, title="GNN-RWKV Session", border_style="cyan"))
    visible = messages[-10:]
    if not visible:
        body.append(Panel("Brain ready. Type below to chat.", border_style="dim"))
    for message in visible:
        style = "bright_cyan" if message.role == "You" else "bright_green"
        body.append(Panel(Text(message.text, style="white"), title=message.role, border_style=style))
    body.append(Panel("/train <topic/url>  /retrain  /clear  quit", title="Commands", border_style="magenta"))
    return Group(*body)


def load_active_brain_paths():
    if os.path.exists(ACTIVE_BRAIN_PATH):
        try:
            with open(ACTIVE_BRAIN_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            weights = manifest.get("weights", WEIGHTS_PATH)
            vocab = manifest.get("vocab", VOCAB_PATH)
            meta = manifest.get("meta", META_PATH)
            return weights, vocab, meta
        except (OSError, json.JSONDecodeError):
            pass
    return WEIGHTS_PATH, VOCAB_PATH, META_PATH


def is_saved_brain_ready():
    weights, vocab, meta_path = load_active_brain_paths()
    if not (os.path.exists(weights) and os.path.exists(vocab) and os.path.exists(meta_path)):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return meta.get("tokenizer") == TOKENIZER_VERSION


def train_startup_brain(data_file=DEFAULT_DATA_FILE, epochs=8, max_sentences=DEFAULT_MAX_SENTENCES):
    from train import train_brain

    status = {"text": f"First run: training from {data_file} ({max_sentences} lines)."}

    def progress(epoch, total, loss):
        status["text"] = f"One-time training {epoch}/{total} | loss {loss:.4f}"

    with Live(render_chat([], status["text"]), console=console, refresh_per_second=6) as live:
        def live_progress(epoch, total, loss):
            progress(epoch, total, loss)
            live.update(render_chat([], status["text"]))

        train_brain(
            data_file,
            epochs=epochs,
            persist=True,
            max_sentences=max_sentences,
            progress_callback=live_progress,
            verbose=False,
            reset_existing=True,
        )
        live.update(render_chat([], "Training saved. Future restarts load social_brain.pth directly."))


async def start_chat():
    interactive_terminal = sys.stdin.isatty() and sys.stdout.isatty() and PromptSession is not None
    session = PromptSession() if interactive_terminal else None
    messages = []

    if not is_saved_brain_ready():
        train_startup_brain()

    weights_path, vocab_path, _ = load_active_brain_paths()
    brain = PersistentGNNBrain(weights_path, vocab_path)
    status = "Ready. Loaded saved brain. It will not retrain on every restart."

    while True:
        try:
            if interactive_terminal:
                console.clear()
            console.print(render_chat(messages, status))
            if session:
                u_input = (await session.prompt_async(HTML("<ansicyan>You</ansicyan> > "))).strip()
            else:
                u_input = (await asyncio.to_thread(input, "You > ")).strip()
            if u_input.lower() in ['quit', 'exit']: break
            if u_input == "/clear":
                messages.clear()
                continue
            if u_input == "/retrain":
                status = "Retraining saved brain from local 1000-line corpus..."
                train_startup_brain()
                weights_path, vocab_path, _ = load_active_brain_paths()
                brain = PersistentGNNBrain(weights_path, vocab_path)
                status = "Retrained and saved. Future restarts load this brain."
                continue
            if u_input.startswith("/train"):
                query = u_input.replace("/train", "").strip()
                messages.append(ChatMessage("You", u_input))
                await crawl_and_train(query, brain)
                status = "Live web training finished and brain reloaded."
                continue
            if not u_input: continue
            messages.append(ChatMessage("You", u_input))
            ans = brain.chat(u_input)
            messages.append(ChatMessage("Brain", ans))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(start_chat())
