import torch
import math

def get_rotary_angles(seq_len, head_dim, base=10000):
    # Compute rotary angles
    angles = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    # Outer product: [seq_len, head_dim // 2]
    angle_rads = torch.einsum('n,d->nd', pos, angles)  # (seq_len, head_dim // 2)
    # print('Angle rads: ', angle_rads)
    return angle_rads

def apply_rotary_emb(x, angle_rads):
    # x: [batch, seq_len, num_heads, head_dim]
    # angle_rads: [seq_len, head_dim/2]
    cos = angle_rads.cos()
    sin = angle_rads.sin()
    head_dim = x.shape[-1]
    # Reshape for broadcasting over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(2)     # [1, seq_len, 1, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    # Split last dim into [even, odd]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Apply rotation
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return x_rot

# Example usage:
B, S, H, D = 2, 8, 4, 64        # Batch, Seq, Heads, HeadDim
x = torch.randn(B, S, H, D)     # Query or Key tensor in attention
angle_rads = get_rotary_angles(S, D)
x_rope = apply_rotary_emb(x, angle_rads)   # Use for attention

print("Pre-RoPE shape:", x.shape)
print("After RoPE shape:", x_rope.shape)
