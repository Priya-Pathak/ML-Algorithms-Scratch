import torch

class KVCache:
    def __init__(self):
        # Initialize empty key and value caches
        self.cache_k = None
        self.cache_v = None

    def append(self, k_new, v_new):
        # k_new, v_new: [batch, heads, seq(=1), head_dim]
        # On first token, initialize; else, append along seq dimension
        if self.cache_k is None:
            self.cache_k = k_new
            self.cache_v = v_new
        else:
            self.cache_k = torch.cat([self.cache_k, k_new], dim=2)  # concat on seq axis
            self.cache_v = torch.cat([self.cache_v, v_new], dim=2)

    def get(self):
        # Return cached tensors
        return self.cache_k, self.cache_v

    def reset(self):
        self.cache_k = None
        self.cache_v = None

# Example usage in a toy inference loop:
B, H, D = 1, 4, 32   # batch, heads, head_dim
cache = KVCache()
for step in range(5):  # simulate 5 decoding steps
    k = torch.randn(B, H, 1, D)
    v = torch.randn(B, H, 1, D)
    cache.append(k, v)
    kcat, vcat = cache.get()
    print(f"Step {step}: Cached KV shapes: {kcat.shape}, {vcat.shape}")
# Output will increment seq_len each time: (B, H, step+1, D)

# Reset cache when starting new sequence
cache.reset()
