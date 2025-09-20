import torch

def sinusoidal_positional_embedding(seq_len, d_model, base=10000):
    """
    Args:
      seq_len:   Number of positions (e.g., max sequence length, int)
      d_model:   Embedding dimension (must be even, int)
      base:      The wavelength base used in the original paper (float)
    Returns:
      pe:        [seq_len, d_model] positional embedding matrix (no batch dim)
    """
    # 1. [seq_len, 1] column vector with each position [0, 1, ..., seq_len-1]
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    # 2. [1, d_model//2] row vector: Compute the scaling factors
    div_term = base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)  # shape [d_model//2]

    # 3. [seq_len, d_model] Preallocate
    pe = torch.zeros(seq_len, d_model)
    # 4. Fill even indices with sin, odd indices with cos
    pe[:, 0::2] = torch.sin(positions / div_term)
    pe[:, 1::2] = torch.cos(positions / div_term)
    return pe

# Usage:
seq_len, d_model = 10, 8
pos_emb = sinusoidal_positional_embedding(seq_len, d_model)
print("Output shape:", pos_emb.shape)
print(pos_emb)
