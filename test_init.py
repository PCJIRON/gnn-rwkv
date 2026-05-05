import torch
from gnn_rwkv_story_gen import SocialNarrativeModel
import time

v_size = 26072
dim = 256
print(f"Testing model init with vocab={v_size}, dim={dim}...")
start = time.time()
model = SocialNarrativeModel(v_size, 5, dim)
print(f"Model init done in {time.time()-start:.2f}s")

print("Listing parameters...")
start = time.time()
params = list(model.parameters())
print(f"Got {len(params)} parameter groups in {time.time()-start:.2f}s")

print("Testing optimizer init...")
start = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print(f"Optimizer init done in {time.time()-start:.2f}s")
