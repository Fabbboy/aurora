import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        # Split the embedding dimension into multiple heads
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x, batch_size):
        # Combine the heads back into the original embedding dimension
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = torch.clamp(scores, min=-1e9, max=1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = torch.softmax(scores, dim=-1) + 1e-9

        return attn @ v

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = [self.split_heads(t, batch_size) for t in (q, k, v)]
        attn = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = self.combine_heads(attn, batch_size)
        output = self.out_proj(attn_output)
        return output

def test_multi_head_attention():
    embed_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(1, 40, embed_dim)
    out = mha(x)
    assert out.shape == (1, 40, embed_dim), f"Wrong shape: {out.shape}"
    print("MultiHeadAttention test passed.")

if __name__ == "__main__":
    test_multi_head_attention()
